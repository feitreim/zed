//! Render-cost benchmarks for the agent panel while an agent is answering.
//!
//! These reproduce the workload behind
//! <https://github.com/zed-industries/zed/issues/51597>, where Zed's energy and
//! GPU use spike for the whole duration of an agent turn. Two things redraw the
//! window while a turn is in flight, and the benchmarks separate them:
//!
//! * The "generating" spinner is a repeating animation. It used to ask for a
//!   new frame every vsync for as long as the turn lasted; it now only
//!   invalidates the window when it advances to its next glyph
//!   (`agent_animation_frame` measures the per-vsync cost either way).
//! * Streamed assistant text is appended to a `Markdown` entity, which reparses
//!   and redraws (`agent_stream_chunk`).
//!
//! `agent_generating_second` combines both at the rate they occur in
//! production, so its per-iteration time is a proxy for the foreground work one
//! second of generation costs.
//!
//! The agent is a `StubAgentConnection`, so nothing here depends on a
//! particular provider: this is the path every agent's output takes, native or
//! external/ACP.
//!
//! Setup asserts how many of `IDLE_FRAMES` vsyncs actually draw the window, so
//! a benchmark can't quietly start measuring a panel that has stopped
//! redrawing (or one that never started).

use std::{rc::Rc, sync::Arc};

use acp_thread::{
    AcpThread, AgentThreadEntry, AssistantMessageChunk, ContentBlock, StubAgentConnection,
    ThreadStatus,
};
use agent_client_protocol::schema::v1 as acp;
use agent_ui::AgentPanel;
use agent_ui::test_support::{StubAgentServer, init_globals, send_message_with_text};
use clock::FakeSystemClock;
use editor::{Editor, EditorMode, MultiBuffer};
use gpui::{AnyWindowHandle, App, AppContext as _, BenchAppContext, Entity, Window, profiler};
use markdown::Markdown;
use project::{LocalProjectFlags, Project};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use workspace::MultiWorkspace;

/// Frames in one simulated second, matching a ProMotion display. The spinner
/// requests a frame per vsync, so this is also the redraw count the animation
/// forces per second of generation.
const FRAMES_PER_SECOND: usize = 120;

/// How often the assistant message grows, in frames. `AcpThread` reveals
/// buffered agent text on a 16ms timer, i.e. roughly every other frame at
/// 120Hz.
const FRAMES_PER_CHUNK: usize = 2;

/// A chunk of revealed text, sized like one 16ms slice of a streamed response.
const CHUNK: &str = "the quick brown fox jumps over the lazy dog. ";

/// The assistant message the thread starts each benchmark with, so streaming
/// measurements include the cost of reparsing and relaying out a
/// partially-written response rather than an empty one.
const INITIAL_RESPONSE_BYTES: usize = 4096;

/// Frames used to settle the UI, and to check how often a generating panel
/// redraws when nothing about the conversation changed.
const IDLE_FRAMES: usize = 10;

#[gpui::bench(
    inputs = layouts(),
    group = "Agent animation frame",
    input_name = "layout",
    sample_size = 20
)]
fn agent_animation_frame(layout: &&str, cx: &mut BenchAppContext) {
    let harness = Harness::setup(Layout::from_name(layout), cx);
    harness.begin_turn(cx);

    cx.bench_iter(|cx| {
        harness.simulate_frame(cx);
    });

    harness.shutdown(cx);
}

#[gpui::bench(
    inputs = layouts(),
    group = "Agent stream chunk",
    input_name = "layout",
    sample_size = 20
)]
fn agent_stream_chunk(layout: &&str, cx: &mut BenchAppContext) {
    let harness = Harness::setup(Layout::from_name(layout), cx);
    harness.begin_turn(cx);
    harness.stream_initial_response(cx);

    cx.bench_iter(|cx| {
        harness.reveal_chunk(CHUNK, cx);
        harness.simulate_frame(cx);
    });

    harness.shutdown(cx);
}

#[gpui::bench(
    inputs = layouts(),
    group = "Agent generating second",
    input_name = "layout",
    sample_size = 10
)]
fn agent_generating_second(layout: &&str, cx: &mut BenchAppContext) {
    let harness = Harness::setup(Layout::from_name(layout), cx);
    harness.begin_turn(cx);
    harness.stream_initial_response(cx);

    cx.bench_iter(|cx| {
        for frame in 0..FRAMES_PER_SECOND {
            if frame % FRAMES_PER_CHUNK == 0 {
                harness.reveal_chunk(CHUNK, cx);
            }
            harness.simulate_frame(cx);
        }
    });

    harness.shutdown(cx);
}

/// One second of generation as seen by `AcpThread`: chunks arrive over the
/// wire and are revealed by the thread's own timer, so this includes the ACP
/// plumbing (entry updates, list remeasures, view state sync) that
/// `agent_generating_second` skips in order to stay deterministic.
#[gpui::bench(
    inputs = layouts(),
    group = "Agent generating second (over ACP)",
    input_name = "layout",
    sample_size = 10
)]
fn agent_generating_second_over_acp(layout: &&str, cx: &mut BenchAppContext) {
    let harness = Harness::setup(Layout::from_name(layout), cx);
    harness.begin_turn(cx);
    harness.stream_initial_response(cx);

    cx.bench_iter(|cx| {
        for frame in 0..FRAMES_PER_SECOND {
            if frame % FRAMES_PER_CHUNK == 0 {
                harness.send_chunk(CHUNK, cx);
            }
            harness.simulate_frame(cx);
        }
        // Drain the reveal timer's work so a run's appends are measured by the
        // run that produced them.
        cx.run_until_idle();
    });

    harness.shutdown(cx);
}

#[derive(Clone, Copy, PartialEq)]
struct Layout {
    /// Whether a full-size editor is open in the center pane. The center pane
    /// is a cached view, so this shows how far a redraw triggered by the agent
    /// panel reaches into the rest of the window.
    center_editor: bool,
    /// Whether `reduce_motion` is on, which stops the spinner from scheduling
    /// animation frames. This is the workaround available to users today, and
    /// measuring it separates spinner cost from streaming cost.
    reduce_motion: bool,
}

impl Layout {
    fn from_name(name: &str) -> Self {
        match name {
            "panel_only" => Self {
                center_editor: false,
                reduce_motion: false,
            },
            "with_editor" => Self {
                center_editor: true,
                reduce_motion: false,
            },
            "with_editor_reduced_motion" => Self {
                center_editor: true,
                reduce_motion: true,
            },
            other => panic!("unknown layout {other}"),
        }
    }
}

fn layouts() -> Vec<&'static str> {
    vec!["panel_only", "with_editor", "with_editor_reduced_motion"]
}

struct Harness {
    window: AnyWindowHandle,
    panel: Entity<AgentPanel>,
    connection: StubAgentConnection,
    session_id: acp::SessionId,
    layout: Layout,
}

impl Harness {
    fn setup(layout: Layout, cx: &mut BenchAppContext) -> Self {
        let fs = fs::FakeFs::new(cx.background_executor().clone());
        cx.update(|cx| {
            init_globals(cx);
            <dyn fs::Fs>::set_global(fs.clone(), cx);
            cx.set_reduce_motion(layout.reduce_motion);
            assets::Assets.load_test_fonts(cx);
            prompt_store::init(cx);
            agent::ThreadStore::init_global(cx);
            agent_ui::thread_metadata_store::ThreadMetadataStore::init_global(cx);
            language_model::LanguageModelRegistry::test(cx);
        });

        let languages = Arc::new(language::LanguageRegistry::test(
            cx.background_executor().clone(),
        ));
        let project = cx.update(|cx| {
            let client = client::Client::new(
                Arc::new(FakeSystemClock::new()),
                http_client::FakeHttpClient::with_404_response(),
                cx,
            );
            let user_store = cx.new(|cx| client::UserStore::new(client.clone(), cx));
            Project::local(
                client,
                node_runtime::NodeRuntime::unavailable(),
                user_store,
                languages,
                fs.clone(),
                None,
                LocalProjectFlags {
                    // Nothing here touches the filesystem, and worktree
                    // scanning would need to be awaited.
                    init_worktree_trust: false,
                    watch_global_configs: false,
                },
                cx,
            )
        });

        let mut window = cx.add_empty_window();
        let window_handle = window.window_handle();
        let workspace = window.update(|window, cx| {
            let multi_workspace = window.replace_root(cx, |window, cx| {
                MultiWorkspace::test_new(project.clone(), window, cx)
            });
            multi_workspace.read(cx).workspace().clone()
        });

        if layout.center_editor {
            window.update(|window, cx| {
                let buffer = MultiBuffer::build_simple(&source_file(), cx);
                let editor =
                    cx.new(|cx| Editor::new(EditorMode::full(), buffer, None, window, cx));
                workspace.update(cx, |workspace, cx| {
                    workspace.add_item_to_active_pane(Box::new(editor), None, false, window, cx);
                });
            });
        }

        let panel = window.update(|window, cx| {
            workspace.update(cx, |workspace, cx| {
                let panel = cx.new(|cx| AgentPanel::test_new(workspace, window, cx));
                workspace.add_panel(panel.clone(), window, cx);
                workspace.open_panel::<AgentPanel>(window, cx);
                panel
            })
        });
        cx.run_until_idle();

        let connection = StubAgentConnection::new();
        window.update(|window, cx| {
            panel.update(cx, |panel, cx| {
                panel.open_external_thread_with_server(
                    Rc::new(StubAgentServer::new(connection.clone())),
                    window,
                    cx,
                );
            });
        });
        cx.run_until_idle();

        let session_id = cx
            .read(|cx| {
                panel
                    .read(cx)
                    .active_agent_thread(cx)
                    .map(|thread| thread.read(cx).session_id().clone())
            })
            .expect("panel has no active thread");

        Self {
            window: window_handle,
            panel,
            connection,
            session_id,
            layout,
        }
    }

    /// Sends a user message. The stub never resolves the turn, so the thread
    /// stays in the generating state (spinner running) for the whole benchmark.
    fn begin_turn(&self, cx: &mut BenchAppContext) {
        self.update_window(cx, |window, cx| {
            send_message_with_text(&self.panel, "Explain this code", window, cx);
        });
        cx.run_until_idle();
        self.send_chunk("Here is what the code does.\n\n", cx);
        cx.run_until_idle();

        // A benchmark that measured an idle panel would look great and mean
        // nothing, so fail loudly instead.
        let status = cx.read(|cx| {
            self.panel
                .read(cx)
                .active_agent_thread(cx)
                .expect("panel has no active thread")
                .read(cx)
                .status()
        });
        assert_eq!(
            status,
            ThreadStatus::Generating,
            "thread should be generating for the whole benchmark"
        );

        // Settle transient animations before measuring: the streamed chunk
        // above re-shows the scrollbar, whose fade is a real-time animation
        // that redraws every frame until it completes, so give it wall-clock
        // time to finish. With motion enabled, the spinner's own stepped
        // redraws (at most one per ~57-100ms glyph interval) can still land
        // in a burst, hence a threshold of 1 instead of 0.
        let settled_threshold = if self.layout.reduce_motion { 0 } else { 1 };
        let mut draws = usize::MAX;
        for _ in 0..80 {
            draws = self.count_draws(IDLE_FRAMES, cx);
            if draws <= settled_threshold {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        if self.layout.reduce_motion {
            assert_eq!(
                draws, 0,
                "a generating panel redrew {draws} times in {IDLE_FRAMES} frames \
                 with no new content and reduce_motion on"
            );
        } else {
            // The spinner is a stepped animation: it only invalidates the
            // window when it advances to its next glyph (every ~57-100ms), so
            // back-to-back frames with no new content shouldn't draw.
            assert!(
                draws <= 1,
                "a generating panel redrew {draws} times in {IDLE_FRAMES} \
                 back-to-back frames; the spinner should only redraw when its \
                 glyph advances"
            );
            // But it must still be animating: once the spinner's glyph
            // interval has elapsed, the next frame has to redraw.
            std::thread::sleep(std::time::Duration::from_millis(110));
            let draws = self.count_draws(2, cx);
            assert!(
                draws >= 1,
                "the generating spinner should redraw once its glyph interval \
                 has elapsed"
            );
        }
    }

    /// Runs `frames` vsyncs with no new content and returns how many of them
    /// actually drew the window.
    fn count_draws(&self, frames: usize, cx: &mut BenchAppContext) -> usize {
        let was_enabled = profiler::set_frame_trace_enabled(true);
        let mut collector = profiler::FrameTimingCollector::new();
        for _ in 0..frames {
            self.simulate_frame(cx);
        }
        let window_id = self.window.window_id();
        let draws = collector
            .collect_unseen()
            .iter()
            .filter(|timing| timing.window_id == window_id)
            .count();
        if was_enabled {
            profiler::set_frame_trace_enabled(false);
        }
        draws
    }

    /// Grows the assistant message to a realistic size before measuring, since
    /// both reparsing and layout scale with the length of the message being
    /// streamed into.
    fn stream_initial_response(&self, cx: &mut BenchAppContext) {
        let mut streamed = 0;
        while streamed < INITIAL_RESPONSE_BYTES {
            self.reveal_chunk(CHUNK, cx);
            streamed += CHUNK.len();
        }
        cx.run_until_idle();
    }

    /// Delivers a chunk the way an agent does, leaving it to `AcpThread`'s
    /// reveal timer to move it into the rendered message.
    fn send_chunk(&self, text: &str, cx: &mut BenchAppContext) {
        cx.update(|cx| {
            self.connection.send_update(
                self.session_id.clone(),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(text.into())),
                cx,
            );
        });
    }

    /// Appends directly to the streaming message, which is what `AcpThread`'s
    /// reveal timer does on each 16ms tick. Doing it inline keeps the
    /// measurement deterministic: no real time has to pass for the text to
    /// appear.
    fn reveal_chunk(&self, text: &str, cx: &mut BenchAppContext) {
        let markdown = self.streaming_markdown(cx);
        cx.update(|cx| {
            markdown.update(cx, |markdown, cx| markdown.append(text, cx));
        });
        // The reparse runs in the background; wait for it so the redraw it
        // triggers is attributed to this chunk.
        cx.run_until_idle();
    }

    fn streaming_markdown(&self, cx: &BenchAppContext) -> Entity<Markdown> {
        cx.read(|cx| {
            let thread = self
                .panel
                .read(cx)
                .active_agent_thread(cx)
                .expect("panel has no active thread");
            last_assistant_markdown(&thread, cx)
        })
    }

    /// Runs one vsync: delivers the frame callbacks the spinner's animation
    /// registered, draws the window if anything invalidated it, and submits the
    /// scene.
    fn simulate_frame(&self, cx: &mut BenchAppContext) {
        self.update_window(cx, |window, cx| {
            window.simulate_next_frame(cx);
        });
        self.update_window(cx, |window, _| window.present_if_needed());
    }

    /// Closes the window and lets the thread's timer-driven tasks (the
    /// streaming reveal, git polling, and friends) release the entities they
    /// captured, so GPUI's leak detector doesn't fire when the benchmark app is
    /// torn down.
    fn shutdown(self, cx: &mut BenchAppContext) {
        let window = self.window;
        drop(self);
        cx.update_window(window, |_, window, _| window.remove_window())
            .ok();
        cx.run_until_idle();

        let dispatcher = cx.background_executor().dispatcher().clone();
        let dispatcher = dispatcher.as_bench().expect("bench dispatcher");
        for _ in 0..100 {
            if dispatcher.cancel_pending_timers() == 0 {
                break;
            }
            cx.run_until_idle();
        }
    }

    fn update_window<R>(
        &self,
        cx: &mut BenchAppContext,
        update: impl FnOnce(&mut Window, &mut App) -> R,
    ) -> R {
        cx.update_window(self.window, |_, window, cx| update(window, cx))
            .expect("benchmark window was closed")
    }
}

fn last_assistant_markdown(thread: &Entity<AcpThread>, cx: &App) -> Entity<Markdown> {
    let Some(AgentThreadEntry::AssistantMessage(message)) = thread.read(cx).entries().last() else {
        panic!("thread's last entry is not an assistant message");
    };
    let Some(
        AssistantMessageChunk::Message {
            block: ContentBlock::Markdown { markdown },
            ..
        }
        | AssistantMessageChunk::Thought {
            block: ContentBlock::Markdown { markdown },
            ..
        },
    ) = message.chunks.last()
    else {
        panic!("assistant message has no markdown chunk to stream into");
    };
    markdown.clone()
}

/// A file for the center pane, sized so that laying the editor out is not free.
fn source_file() -> String {
    let mut rng = StdRng::seed_from_u64(1);
    let mut text = String::new();
    for line in 0..2000 {
        let indent = "    ".repeat(rng.random_range(0..3));
        text.push_str(&format!("{indent}let value_{line} = compute(line_{line});\n"));
    }
    text
}

gpui::bench_group!(
    benches,
    agent_animation_frame,
    agent_stream_chunk,
    agent_generating_second,
    agent_generating_second_over_acp
);
gpui::bench_main!(benches);
