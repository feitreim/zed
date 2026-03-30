use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use editor::MultiBuffer;
use editor::display_map::*;
use gpui::{SharedString, TestDispatcher};
use multi_buffer::{MultiBufferOffset, ToOffset};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use text::Bias;

// Each line has 2 "!=" at known positions, giving predictable concealment density.
const LINE: &str = "if x != y && a != b { z }\n";

fn build_concealments(
    text: &str,
    buffer_snapshot: &multi_buffer::MultiBufferSnapshot,
) -> Vec<(std::ops::Range<multi_buffer::Anchor>, SharedString)> {
    text.match_indices("!=")
        .map(|(i, _)| {
            (
                buffer_snapshot.anchor_after(MultiBufferOffset(i))
                    ..buffer_snapshot.anchor_before(MultiBufferOffset(i + 2)),
                SharedString::from("≠"),
            )
        })
        .collect()
}

fn set_concealments_bench(c: &mut Criterion) {
    let dispatcher = TestDispatcher::new(1);
    let cx = gpui::TestAppContext::build(dispatcher, None);
    let mut group = c.benchmark_group("conceal/set_concealments");

    for num_lines in [100, 1_000, 10_000] {
        let text = LINE.repeat(num_lines);
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let buffer_snapshot = cx.read(|cx| buffer.read(cx).snapshot(cx));
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = FoldMap::new(inlay_snapshot);
        let concealments = build_concealments(&text, &buffer_snapshot);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_lines),
            &num_lines,
            |bench, _| {
                bench.iter_batched(
                    || {
                        let (map, _) = ConcealMap::new(fold_snapshot.clone());
                        (map, concealments.clone())
                    },
                    |(mut map, concealments)| black_box(map.set_concealments(concealments)),
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn to_conceal_point_bench(c: &mut Criterion) {
    let dispatcher = TestDispatcher::new(1);
    let cx = gpui::TestAppContext::build(dispatcher, None);
    let mut group = c.benchmark_group("conceal/to_conceal_point");

    for num_lines in [100, 1_000, 10_000] {
        let text = LINE.repeat(num_lines);
        let text_len = text.len();
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let buffer_snapshot = cx.read(|cx| buffer.read(cx).snapshot(cx));
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = FoldMap::new(inlay_snapshot.clone());

        let mut rng = StdRng::seed_from_u64(42);
        let fold_points: Vec<FoldPoint> = (0..100)
            .map(|_| {
                let offset = rng.random_range(0..text_len);
                fold_snapshot.to_fold_point(
                    inlay_snapshot.to_point(InlayOffset(MultiBufferOffset(offset))),
                    Bias::Left,
                )
            })
            .collect();

        // Passthrough: no concealments, exercises the fast path
        let (_, passthrough) = ConcealMap::new(fold_snapshot.clone());
        group.bench_with_input(
            BenchmarkId::new("passthrough", num_lines),
            &fold_points,
            |bench, points| {
                bench.iter(|| {
                    for &p in points {
                        black_box(passthrough.to_conceal_point(p, Bias::Left));
                    }
                });
            },
        );

        // Active: concealments present, exercises SumTree lookups
        let concealments = build_concealments(&text, &buffer_snapshot);
        let (mut map, _) = ConcealMap::new(fold_snapshot.clone());
        let (active, _) = map.set_concealments(concealments);
        group.bench_with_input(
            BenchmarkId::new("active", num_lines),
            &fold_points,
            |bench, points| {
                bench.iter(|| {
                    for &p in points {
                        black_box(active.to_conceal_point(p, Bias::Left));
                    }
                });
            },
        );
    }
    group.finish();
}

fn chunk_iteration_bench(c: &mut Criterion) {
    let dispatcher = TestDispatcher::new(1);
    let cx = gpui::TestAppContext::build(dispatcher, None);
    let mut group = c.benchmark_group("conceal/chunks");

    for num_lines in [100, 1_000, 10_000] {
        let text = LINE.repeat(num_lines);
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let buffer_snapshot = cx.read(|cx| buffer.read(cx).snapshot(cx));
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = FoldMap::new(inlay_snapshot);

        let (_, passthrough) = ConcealMap::new(fold_snapshot.clone());
        group.bench_with_input(
            BenchmarkId::new("passthrough", num_lines),
            &num_lines,
            |bench, _| {
                bench.iter(|| {
                    black_box(passthrough.chars_at(ConcealPoint::new(0, 0)).count());
                });
            },
        );

        let concealments = build_concealments(&text, &buffer_snapshot);
        let (mut map, _) = ConcealMap::new(fold_snapshot.clone());
        let (active, _) = map.set_concealments(concealments);
        group.bench_with_input(
            BenchmarkId::new("active", num_lines),
            &num_lines,
            |bench, _| {
                bench.iter(|| {
                    black_box(active.chars_at(ConcealPoint::new(0, 0)).count());
                });
            },
        );
    }
    group.finish();
}

fn reveal_sync_bench(c: &mut Criterion) {
    let dispatcher = TestDispatcher::new(1);
    let cx = gpui::TestAppContext::build(dispatcher, None);
    let mut group = c.benchmark_group("conceal/reveal_sync");

    for num_lines in [100, 1_000, 10_000] {
        let text = LINE.repeat(num_lines);
        let mid = text.len() / 2;
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let buffer_snapshot = cx.read(|cx| buffer.read(cx).snapshot(cx));
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = FoldMap::new(inlay_snapshot);
        let concealments = build_concealments(&text, &buffer_snapshot);

        // Reveal concealments near the middle — simulates cursor landing on them
        let mid_end = mid + LINE.len();
        let reveal_indices: Vec<usize> = concealments
            .iter()
            .enumerate()
            .filter(|(_, (range, _))| {
                let start = range.start.to_offset(&buffer_snapshot);
                let end = range.end.to_offset(&buffer_snapshot);
                end > MultiBufferOffset(mid) && start < MultiBufferOffset(mid_end)
            })
            .map(|(i, _)| i)
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_lines),
            &num_lines,
            |bench, _| {
                bench.iter_batched(
                    || {
                        let (mut map, _) = ConcealMap::new(fold_snapshot.clone());
                        let _ = map.set_concealments(concealments.clone());
                        map
                    },
                    |mut map| {
                        map.set_revealed_indices(reveal_indices.clone());
                        black_box(map.read(fold_snapshot.clone(), vec![]))
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    set_concealments_bench,
    to_conceal_point_bench,
    chunk_iteration_bench,
    reveal_sync_bench,
);
criterion_main!(benches);
