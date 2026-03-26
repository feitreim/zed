// ConcealMap: a display pipeline layer that visually replaces text ranges with
// shorter substitutes (e.g. "lambda" → "λ"). It sits between FoldMap and TabMap:
//
//   InlayMap → FoldMap → **ConcealMap** → TabMap → WrapMap → BlockMap
//
// Like FoldMap, it uses a SumTree<Transform> where each node is either:
//   - Isomorphic: input passes through unchanged (input == output summary)
//   - Concealment: input text is replaced with a shorter string (input ≠ output)
//
// Concealments are stored as buffer Anchors so they survive edits. On each sync,
// anchors are resolved to FoldOffsets and the transform tree is rebuilt.
//
// Revealed ranges (driven by cursor position) suppress concealments on those lines,
// letting the user see and edit the original text.

use super::{
    Highlights,
    fold_map::{Chunk, FoldChunks, FoldEdit, FoldOffset, FoldPoint, FoldRows, FoldSnapshot},
};
use gpui::SharedString;
use language::{LanguageAwareStyling, Point};
use multi_buffer::{Anchor, MBTextSummary, MultiBufferOffset, RowInfo, ToOffset};
use std::{
    cmp,
    ops::{Add, AddAssign, Deref, Range, Sub, SubAssign},
};
use sum_tree::{Bias, Cursor, Dimensions, SumTree};


/// Mutable state for the conceal layer. Holds the current snapshot plus the
/// concealment definitions and reveal state that drive the next rebuild.
pub struct ConcealMap {
    snapshot: ConcealSnapshot,
    /// The active concealments: each is a buffer anchor range (what to hide)
    /// paired with a replacement string (what to show instead).
    concealments: Vec<(Range<Anchor>, SharedString)>,
    /// Buffer ranges where concealments are suppressed (typically the cursor's line).
    /// Any concealment overlapping a revealed range is skipped during build_transforms.
    revealed_ranges: Vec<Range<Anchor>>,
    /// When true, revealed_ranges changed but the transform tree hasn't been rebuilt yet.
    /// The next sync() call will pick this up and rebuild.
    revealed_dirty: bool,
}

/// Immutable view of the conceal layer at a point in time. Derefs to
/// FoldSnapshot so callers can transparently access fold/inlay/buffer data.
/// The transforms SumTree is the core data structure mapping between
/// fold-space (input) and conceal-space (output).
#[derive(Clone)]
pub struct ConcealSnapshot {
    pub fold_snapshot: FoldSnapshot,
    transforms: SumTree<Transform>,
    /// Monotonically increasing version. Downstream layers (TabMap, WrapMap)
    /// compare this to detect when they need to re-sync.
    pub version: usize,
}

/// Deref to FoldSnapshot lets ConcealSnapshot transparently expose all
/// fold/inlay/buffer methods. ConcealSnapshot adds conceal-specific methods.
impl Deref for ConcealSnapshot {
    type Target = FoldSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.fold_snapshot
    }
}

// --- Transform types ---
// Each node in the SumTree is a Transform. When replacement is None, it's
// isomorphic (passthrough). When replacement is Some, the input text is
// replaced with the replacement string in the display output.

#[derive(Clone, Debug)]
enum Transform {
    Isomorphic(MBTextSummary),
    Replacement {
        input: MBTextSummary,
        text: SharedString,
    },
}

impl Transform {
    fn is_concealment(&self) -> bool {
        matches!(self, Transform::Replacement { .. })
    }

    fn replacement_text(&self) -> Option<&SharedString> {
        match self {
            Transform::Replacement { text, .. } => Some(text),
            _ => None,
        }
    }
}

impl sum_tree::Item for Transform {
    type Summary = TransformSummary;

    fn summary(&self, _cx: ()) -> Self::Summary {
        match self {
            Transform::Isomorphic(summary) => TransformSummary {
                input: *summary,
                output: *summary,
            },
            Transform::Replacement { input, text } => TransformSummary {
                input: *input,
                output: MBTextSummary::from(text.as_ref()),
            },
        }
    }
}

/// Tracks both input (fold-space) and output (conceal-space) text summaries.
/// For isomorphic nodes, input == output. For concealments, output is the
/// replacement text's summary (shorter). The SumTree aggregates these, enabling
/// O(log n) coordinate conversion between fold and conceal space.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct TransformSummary {
    input: MBTextSummary,
    output: MBTextSummary,
}

impl sum_tree::ContextLessSummary for TransformSummary {
    fn zero() -> Self {
        Default::default()
    }

    fn add_summary(&mut self, other: &Self) {
        self.input += other.input;
        self.output += other.output;
    }
}

pub type ConcealEdit = text::Edit<ConcealOffset>;

// --- Coordinate types ---
// Each pipeline layer defines its own Point/Offset types so the type system
// prevents accidentally mixing coordinates from different layers.

#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialOrd, PartialEq)]
pub struct ConcealOffset(pub MultiBufferOffset);

impl Add for ConcealOffset {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for ConcealOffset {
    type Output = <MultiBufferOffset as Sub>::Output;
    fn sub(self, rhs: Self) -> Self::Output {
        self.0 - rhs.0
    }
}

impl<T> SubAssign<T> for ConcealOffset
where
    MultiBufferOffset: SubAssign<T>,
{
    fn sub_assign(&mut self, rhs: T) {
        self.0 -= rhs;
    }
}

impl<T> Add<T> for ConcealOffset
where
    MultiBufferOffset: Add<T, Output = MultiBufferOffset>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl AddAssign for ConcealOffset {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<T> AddAssign<T> for ConcealOffset
where
    MultiBufferOffset: AddAssign<T>,
{
    fn add_assign(&mut self, rhs: T) {
        self.0 += rhs;
    }
}

impl<'a> sum_tree::Dimension<'a, TransformSummary> for ConcealOffset {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += summary.output.len;
    }
}

/// A point (row, column) in conceal-output space. After concealment, "lambda"
/// becomes "λ", so column values are shifted relative to FoldPoint.
#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialOrd, PartialEq)]
pub struct ConcealPoint(pub Point);

impl ConcealPoint {
    pub fn new(row: u32, column: u32) -> Self {
        Self(Point::new(row, column))
    }

    pub fn row(self) -> u32 {
        self.0.row
    }

    pub fn column(self) -> u32 {
        self.0.column
    }

    pub fn row_mut(&mut self) -> &mut u32 {
        &mut self.0.row
    }
}

/// SumTree dimension impl: ConcealPoint tracks the *output* side of transforms.
/// This lets the tree efficiently seek by conceal-space coordinates.
impl<'a> sum_tree::Dimension<'a, TransformSummary> for ConcealPoint {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += &summary.output.lines;
    }
}

/// FoldPoint/FoldOffset track the *input* side of transforms, so the tree can
/// also seek by fold-space coordinates. Combined with ConcealPoint/ConcealOffset
/// (output side) via Dimensions<A, B>, this enables bidirectional conversion.
impl<'a> sum_tree::Dimension<'a, TransformSummary> for FoldPoint {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += &summary.input.lines;
    }
}

impl<'a> sum_tree::Dimension<'a, TransformSummary> for FoldOffset {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += summary.input.len;
    }
}

// --- Iterator structs ---

/// Yields text chunks in conceal-output space. Walks the transform tree and the
/// underlying fold chunks in parallel:
/// - For concealment transforms: yields the replacement text with inherited highlighting
/// - For isomorphic transforms: forwards chunks from the fold layer, slicing at
///   transform boundaries
pub struct ConcealChunks<'a> {
    transform_cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealOffset, FoldOffset>>,
    fold_chunks: FoldChunks<'a>,
    /// Cached current fold chunk and its starting offset, to avoid re-seeking
    /// when a fold chunk spans multiple transform boundaries.
    fold_chunk: Option<(FoldOffset, Chunk<'a>)>,
    fold_offset: FoldOffset,
    output_offset: ConcealOffset,
    max_output_offset: ConcealOffset,
    /// Byte offset into the current replacement string (for partial reads).
    replacement_offset: usize,
}

/// Iterates row metadata (soft wrap info, buffer row mapping) in conceal-space.
/// Wraps the underlying FoldRows, skipping rows that are consumed by multi-line
/// concealments (if any — most concealments are single-line).
#[derive(Clone)]
pub struct ConcealRows<'a> {
    cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealPoint, FoldPoint>>,
    input_rows: FoldRows<'a>,
    conceal_point: ConcealPoint,
}

// --- ConcealPoint / ConcealOffset methods ---

impl ConcealPoint {
    /// Maps a conceal-space point back to fold-space. For isomorphic regions this
    /// is a simple offset addition. For concealment regions the point maps to the
    /// start or end of the concealed fold range (depending on bias at call site).
    pub fn to_fold_point(self, snapshot: &ConcealSnapshot) -> FoldPoint {
        let (start, _, _) = snapshot
            .transforms
            .find::<Dimensions<ConcealPoint, FoldPoint>, _>((), &self, Bias::Right);
        let overshoot = self.0 - start.0.0;
        FoldPoint(start.1.0 + overshoot)
    }

    /// Converts a conceal point to a conceal offset by finding the transform node
    /// containing this point and computing the byte offset within it.
    pub fn to_offset(self, snapshot: &ConcealSnapshot) -> ConcealOffset {
        let (start, _, item) = snapshot
            .transforms
            .find::<Dimensions<ConcealPoint, TransformSummary>, _>((), &self, Bias::Right);
        let overshoot = self.0 - start.1.output.lines;
        let mut offset = start.1.output.len;
        if !overshoot.is_zero() {
            if let Some(transform) = item {
                // Must be isomorphic — you can't be "inside" a concealment's
                // output at a row overshoot since replacements are typically
                // single-line and short.
                assert!(!transform.is_concealment());
                let end_fold_offset =
                    FoldPoint(start.1.input.lines + overshoot).to_offset(&snapshot.fold_snapshot);
                offset += end_fold_offset.0 - start.1.input.len;
            } else {
                // Past the end of all transforms — clamp to document end.
                return snapshot.len();
            }
        }
        ConcealOffset(offset)
    }
}

impl ConcealOffset {
    pub fn to_point(self, snapshot: &ConcealSnapshot) -> ConcealPoint {
        let (start, _, item) = snapshot
            .transforms
            .find::<Dimensions<ConcealOffset, TransformSummary>, _>((), &self, Bias::Right);
        if let Some(transform) = item {
            let overshoot = self.0 - start.1.output.len;
            if transform.is_concealment() {
                // Inside a concealment — return start of the concealment in output space
                ConcealPoint(start.1.output.lines)
            } else {
                let fold_offset = FoldOffset(start.1.input.len + overshoot);
                let fold_point = fold_offset.to_point(&snapshot.fold_snapshot);
                let fold_start = FoldPoint(start.1.input.lines);
                ConcealPoint(start.1.output.lines + (fold_point.0 - fold_start.0))
            }
        } else {
            snapshot.max_point()
        }
    }
}

// --- ConcealMap impl ---

impl ConcealMap {
    /// Creates a new passthrough ConcealMap with no concealments. The initial
    /// transform tree is a single isomorphic node spanning all fold output.
    pub fn new(fold_snapshot: FoldSnapshot) -> (Self, ConcealSnapshot) {
        let mut snapshot = ConcealSnapshot {
            transforms: SumTree::default(),
            fold_snapshot,
            version: 0,
        };
        build_transforms(&mut snapshot.transforms, &snapshot.fold_snapshot, &[], &[]);
        (
            Self {
                snapshot: snapshot.clone(),
                concealments: Vec::new(),
                revealed_ranges: Vec::new(),
                revealed_dirty: false,
            },
            snapshot,
        )
    }

    /// Called by the display pipeline during snapshot(). Syncs the conceal layer
    /// with upstream fold changes and returns the updated snapshot + edits for
    /// the next layer (TabMap) to consume.
    pub fn read(
        &mut self,
        fold_snapshot: FoldSnapshot,
        fold_edits: Vec<FoldEdit>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let edits = self.sync(fold_snapshot, fold_edits);
        (self.snapshot.clone(), edits)
    }

    /// Replaces the entire set of concealments and rebuilds the transform tree.
    /// Returns a full-document edit if the output changed, or empty edits if not.
    /// Called by DisplayMap::set_concealments when the editor toggles conceal or
    /// refreshes after a buffer edit.
    pub fn set_concealments(
        &mut self,
        concealments: Vec<(Range<Anchor>, SharedString)>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let old_snapshot = self.snapshot.clone();
        self.concealments = concealments;
        self.snapshot.version += 1;

        let mut new_transforms = SumTree::default();
        build_transforms(
            &mut new_transforms,
            &self.snapshot.fold_snapshot,
            &self.concealments,
            &self.revealed_ranges,
        );

        let old_len = ConcealOffset(old_snapshot.transforms.summary().output.len);
        self.snapshot.transforms = new_transforms;
        let new_len = self.snapshot.len();

        // Only emit an edit if the output actually changed. This prevents
        // unnecessary downstream pipeline work.
        let edits = if old_len == new_len
            && old_snapshot.transforms.summary().output == self.snapshot.transforms.summary().output
        {
            vec![]
        } else {
            vec![ConcealEdit {
                old: ConcealOffset(MultiBufferOffset(0))..old_len,
                new: ConcealOffset(MultiBufferOffset(0))..new_len,
            }]
        };
        (self.snapshot.clone(), edits)
    }

    pub fn concealments(&self) -> &[(Range<Anchor>, SharedString)] {
        &self.concealments
    }

    /// Updates the revealed ranges. The dirty flag is picked up on the next
    /// sync() call (triggered by the render cycle taking a snapshot), which
    /// rebuilds the transform tree excluding concealments on revealed lines.
    pub fn set_revealed_ranges(&mut self, revealed_ranges: Vec<Range<Anchor>>) {
        self.revealed_ranges = revealed_ranges;
        self.revealed_dirty = true;
    }

    /// Reconciles the conceal layer with upstream fold changes.
    ///
    /// Three cases:
    /// 1. No fold edits, no reveal change, same version → no-op (fast path)
    /// 2. No fold edits but reveal changed → rebuild, emit full-doc edit only if output differs
    /// 3. Fold edits present (buffer was edited):
    ///    a. No concealments → passthrough: FoldOffset == ConcealOffset, safe to forward edits
    ///    b. Concealments active → emit full-doc edit because fold-offsets and conceal-offsets
    ///       diverge (e.g. "lambda" is 6 bytes in fold space but "λ" is 2 in conceal space).
    ///       Forwarding fold edits as conceal edits would give the wrong ranges to downstream.
    fn sync(&mut self, fold_snapshot: FoldSnapshot, fold_edits: Vec<FoldEdit>) -> Vec<ConcealEdit> {
        let reveal_changed = self.revealed_dirty;
        self.revealed_dirty = false;

        // Fast path: nothing changed upstream and no pending reveal update.
        if fold_edits.is_empty()
            && self.snapshot.fold_snapshot.version == fold_snapshot.version
            && !reveal_changed
        {
            return Vec::new();
        }

        let old_snapshot = self.snapshot.clone();
        self.snapshot.fold_snapshot = fold_snapshot;

        // Always rebuild the full transform tree. Concealments are stored as
        // buffer Anchors, so they automatically track their positions through edits.
        let mut new_transforms = SumTree::default();
        build_transforms(
            &mut new_transforms,
            &self.snapshot.fold_snapshot,
            &self.concealments,
            &self.revealed_ranges,
        );

        let old_output = old_snapshot.transforms.summary().output;
        let new_output = new_transforms.summary().output;
        self.snapshot.transforms = new_transforms;

        if fold_edits.is_empty() {
            // Reveal change only — only bump version if output actually changed,
            // preventing infinite re-render loops.
            if old_output == new_output {
                return vec![];
            }
            self.snapshot.version += 1;
            let old_len = ConcealOffset(old_output.len);
            let new_len = ConcealOffset(new_output.len);
            vec![ConcealEdit {
                old: ConcealOffset(MultiBufferOffset(0))..old_len,
                new: ConcealOffset(MultiBufferOffset(0))..new_len,
            }]
        } else if self.concealments.is_empty() {
            // No concealments — pure passthrough. FoldOffset == ConcealOffset,
            // so we can forward fold edits directly as conceal edits.
            self.snapshot.version += 1;
            fold_edits
                .into_iter()
                .map(|edit| ConcealEdit {
                    old: ConcealOffset(edit.old.start.0)..ConcealOffset(edit.old.end.0),
                    new: ConcealOffset(edit.new.start.0)..ConcealOffset(edit.new.end.0),
                })
                .collect()
        } else {
            // Concealments active + buffer edited. We can't translate fold edits
            // to conceal edits without walking both old and new transform trees,
            // so we conservatively emit a single full-document edit.
            self.snapshot.version += 1;
            let old_len = ConcealOffset(old_output.len);
            let new_len = ConcealOffset(new_output.len);
            vec![ConcealEdit {
                old: ConcealOffset(MultiBufferOffset(0))..old_len,
                new: ConcealOffset(MultiBufferOffset(0))..new_len,
            }]
        }
    }
}

// --- ConcealSnapshot impl ---

impl ConcealSnapshot {
    /// Maps a fold-space point to conceal-space. If the point falls inside a
    /// concealment, it snaps to the start or end depending on bias (Left snaps
    /// to start, Right snaps to end — this controls cursor behavior at boundaries).
    pub fn to_conceal_point(&self, point: FoldPoint, bias: Bias) -> ConcealPoint {
        let (start, end, item) = self
            .transforms
            .find::<Dimensions<FoldPoint, ConcealPoint>, _>((), &point, Bias::Right);
        if item.is_some_and(|t| t.is_concealment()) {
            if bias == Bias::Left || point == start.0 {
                start.1
            } else {
                end.1
            }
        } else {
            let overshoot = point.0 - start.0.0;
            ConcealPoint(cmp::min(start.1.0 + overshoot, end.1.0))
        }
    }

    pub fn len(&self) -> ConcealOffset {
        ConcealOffset(self.transforms.summary().output.len)
    }

    pub fn max_point(&self) -> ConcealPoint {
        ConcealPoint(self.transforms.summary().output.lines)
    }

    pub fn line_len(&self, row: u32) -> u32 {
        let line_start = ConcealPoint::new(row, 0).to_offset(self).0;
        let line_end = if row >= self.max_point().row() {
            self.len().0
        } else {
            ConcealPoint::new(row + 1, 0).to_offset(self).0 - 1
        };
        (line_end - line_start) as u32
    }

    /// Clamps a conceal point to a valid position. Points inside a concealment
    /// snap to its boundary; points in isomorphic regions delegate to fold's clip_point.
    pub fn clip_point(&self, point: ConcealPoint, bias: Bias) -> ConcealPoint {
        let (start, end, item) = self
            .transforms
            .find::<Dimensions<ConcealPoint, FoldPoint>, _>((), &point, Bias::Right);
        if let Some(transform) = item {
            let transform_start = start.0.0;
            if transform.is_concealment() {
                if point.0 == transform_start || matches!(bias, Bias::Left) {
                    ConcealPoint(transform_start)
                } else {
                    ConcealPoint(end.0.0)
                }
            } else {
                let overshoot = point.0 - transform_start;
                let fold_point = FoldPoint(start.1.0 + overshoot);
                let clipped = self.fold_snapshot.clip_point(fold_point, bias);
                ConcealPoint(start.0.0 + (clipped.0 - start.1.0))
            }
        } else {
            self.max_point()
        }
    }

    /// Creates a forward-only cursor for efficient batch FoldPoint→ConcealPoint mapping.
    /// Used by block_map which maps many points in ascending order.
    pub fn conceal_point_cursor(&self) -> ConcealPointCursor<'_> {
        let cursor = self
            .transforms
            .cursor::<Dimensions<FoldPoint, ConcealPoint>>(());
        ConcealPointCursor { cursor }
    }

    pub fn row_infos(&self, start_row: u32) -> ConcealRows<'_> {
        let conceal_point = ConcealPoint::new(start_row, 0);
        let mut cursor = self
            .transforms
            .cursor::<Dimensions<ConcealPoint, FoldPoint>>(());
        cursor.seek(&conceal_point, Bias::Left);
        let overshoot = conceal_point.0 - cursor.start().0.0;
        let fold_point = FoldPoint(cursor.start().1.0 + overshoot);
        ConcealRows {
            cursor,
            input_rows: self.fold_snapshot.row_infos(fold_point.row()),
            conceal_point,
        }
    }

    pub fn chunks_at(&self, start: ConcealPoint) -> ConcealChunks<'_> {
        self.chunks(
            start.to_offset(self)..self.len(),
            LanguageAwareStyling { tree_sitter: false, diagnostics: false },
            Highlights::default(),
        )
    }

    pub fn chars_at(&self, start: ConcealPoint) -> impl '_ + Iterator<Item = char> {
        self.chunks(
            start.to_offset(self)..self.len(),
            LanguageAwareStyling { tree_sitter: false, diagnostics: false },
            Highlights::default(),
        )
        .flat_map(|chunk| chunk.text.chars())
    }

    /// Creates a chunk iterator over the given conceal-offset range. For isomorphic
    /// regions, chunks are forwarded from the fold layer. For concealments, the
    /// replacement text is yielded instead, with syntax highlighting inherited from
    /// the original text at that position.
    pub(crate) fn chunks<'a>(
        &'a self,
        range: Range<ConcealOffset>,
        language_aware: LanguageAwareStyling,
        highlights: Highlights<'a>,
    ) -> ConcealChunks<'a> {
        let mut transform_cursor = self
            .transforms
            .cursor::<Dimensions<ConcealOffset, FoldOffset>>(());
        transform_cursor.seek(&range.start, Bias::Right);

        let fold_start = {
            let overshoot = range.start - transform_cursor.start().0;
            transform_cursor.start().1 + overshoot
        };

        let transform_end = transform_cursor.end();

        let fold_end = if transform_cursor
            .item()
            .is_none_or(|transform| transform.is_concealment())
        {
            fold_start
        } else if range.end < transform_end.0 {
            let overshoot = range.end - transform_cursor.start().0;
            transform_cursor.start().1 + overshoot
        } else {
            transform_end.1
        };

        ConcealChunks {
            transform_cursor,
            fold_chunks: self.fold_snapshot.chunks(
                fold_start..fold_end,
                language_aware,
                highlights,
            ),
            fold_chunk: None,
            fold_offset: fold_start,
            output_offset: range.start,
            max_output_offset: range.end,
            replacement_offset: 0,
        }
    }

    pub fn text_summary_for_range(&self, range: Range<ConcealPoint>) -> MBTextSummary {
        let fold_start = range.start.to_fold_point(self);
        let fold_end = range.end.to_fold_point(self);
        self.fold_snapshot
            .text_summary_for_range(fold_start..fold_end)
    }

    pub fn to_offset(&self, point: ConcealPoint) -> ConcealOffset {
        point.to_offset(self)
    }

    pub fn to_point(&self, offset: ConcealOffset) -> ConcealPoint {
        offset.to_point(self)
    }

    #[cfg(test)]
    pub fn text(&self) -> String {
        self.chunks(
            ConcealOffset(MultiBufferOffset(0))..self.len(),
            LanguageAwareStyling { tree_sitter: false, diagnostics: false },
            Highlights::default(),
        )
        .map(|c| c.text)
        .collect()
    }
}

// --- ConcealPointCursor ---

/// A forward-only cursor for efficient sequential FoldPoint→ConcealPoint mapping.
/// Unlike to_conceal_point() which does a fresh tree search each call, this cursor
/// remembers its position and uses seek_forward for O(log n) amortized traversal.
pub struct ConcealPointCursor<'transforms> {
    cursor: Cursor<'transforms, 'static, Transform, Dimensions<FoldPoint, ConcealPoint>>,
}

impl ConcealPointCursor<'_> {
    /// Maps a FoldPoint to ConcealPoint, advancing the cursor forward. Points must
    /// be supplied in non-decreasing order for correctness.
    pub fn map(&mut self, point: FoldPoint, bias: Bias) -> ConcealPoint {
        let cursor = &mut self.cursor;
        if cursor.did_seek() {
            cursor.seek_forward(&point, Bias::Right);
        } else {
            cursor.seek(&point, Bias::Right);
        }
        if cursor.item().is_some_and(|t| t.is_concealment()) {
            if bias == Bias::Left || point == cursor.start().0 {
                cursor.start().1
            } else {
                cursor.end().1
            }
        } else {
            let overshoot = point.0 - cursor.start().0.0;
            ConcealPoint(cmp::min(cursor.start().1.0 + overshoot, cursor.end().1.0))
        }
    }
}

// --- Iterator impls ---

impl ConcealChunks<'_> {
    pub(crate) fn seek(&mut self, range: Range<ConcealOffset>) {
        self.transform_cursor.seek(&range.start, Bias::Right);

        let fold_start = {
            let overshoot = range.start - self.transform_cursor.start().0;
            self.transform_cursor.start().1 + overshoot
        };

        let transform_end = self.transform_cursor.end();

        let fold_end = if self
            .transform_cursor
            .item()
            .is_none_or(|transform| transform.is_concealment())
        {
            fold_start
        } else if range.end < transform_end.0 {
            let overshoot = range.end - self.transform_cursor.start().0;
            self.transform_cursor.start().1 + overshoot
        } else {
            transform_end.1
        };

        self.fold_chunks.seek(fold_start..fold_end);
        self.fold_chunk = None;
        self.fold_offset = fold_start;
        self.output_offset = range.start;
        self.max_output_offset = range.end;
        self.replacement_offset = 0;
    }
}

impl<'a> Iterator for ConcealChunks<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.output_offset >= self.max_output_offset {
            return None;
        }

        let transform = self.transform_cursor.item()?;

        if let Some(replacement) = transform.replacement_text() {
            // Concealment: yield the replacement text instead of the original.
            let text = &replacement[self.replacement_offset..];

            // Seek the fold chunks to the concealed range to grab the syntax
            // highlighting from the original text. This makes "λ" inherit the
            // highlight color of "lambda".
            let conceal_fold_start = self.transform_cursor.start().1;
            let conceal_fold_end = self.transform_cursor.end().1;
            self.fold_chunks.seek(conceal_fold_start..conceal_fold_end);
            let highlight_chunk = self.fold_chunks.next();

            self.fold_offset = self.transform_cursor.end().1;
            self.output_offset.0 += text.len();
            self.replacement_offset = 0;
            self.fold_chunk.take();
            self.transform_cursor.next();

            return Some(if let Some(source) = highlight_chunk {
                Chunk {
                    text,
                    syntax_highlight_id: source.syntax_highlight_id,
                    highlight_style: source.highlight_style,
                    diagnostic_severity: source.diagnostic_severity,
                    is_unnecessary: source.is_unnecessary,
                    underline: source.underline,
                    ..Default::default()
                }
            } else {
                Chunk {
                    text,
                    ..Default::default()
                }
            });
        }

        // Isomorphic transform: forward the underlying fold chunk.
        // Lazily seek fold_chunks when we don't have a cached chunk.
        if self.fold_chunk.is_none() {
            let transform_end = self.transform_cursor.end();
            let fold_end = if self.max_output_offset < transform_end.0 {
                let overshoot = self.max_output_offset - self.transform_cursor.start().0;
                self.transform_cursor.start().1 + overshoot
            } else {
                transform_end.1
            };
            self.fold_chunks.seek(self.fold_offset..fold_end);
            let chunk_offset = self.fold_offset;
            self.fold_chunk = self.fold_chunks.next().map(|chunk| (chunk_offset, chunk));
        }

        // Slice the fold chunk to fit within the current transform boundary.
        // A single fold chunk may span multiple transforms, so we emit only the
        // portion that belongs to the current one, advancing the transform cursor
        // when we reach its end.
        let (chunk_start, chunk) = self.fold_chunk.clone()?;
        let chunk_end = chunk_start + chunk.text.len();
        let transform_end = self.transform_cursor.end().1;
        let end = chunk_end.min(transform_end);

        let bit_start = self.fold_offset - chunk_start;
        let bit_end = end - chunk_start;
        let text = &chunk.text[bit_start..bit_end];
        // Shift the tab/char/newline bitmasks to match the sliced portion.
        let mask = 1u128.unbounded_shl(bit_end as u32).wrapping_sub(1);
        let tabs = (chunk.tabs >> bit_start) & mask;
        let chars = (chunk.chars >> bit_start) & mask;
        let newlines = (chunk.newlines >> bit_start) & mask;

        if end == transform_end {
            self.transform_cursor.next();
        }
        if end == chunk_end {
            self.fold_chunk.take();
        }

        self.fold_offset = end;
        self.output_offset.0 += text.len();

        debug_assert!(
            !text.is_empty(),
            "empty chunk at fold_offset {:?}",
            self.fold_offset
        );

        Some(Chunk {
            text,
            tabs,
            chars,
            newlines,
            syntax_highlight_id: chunk.syntax_highlight_id,
            highlight_style: chunk.highlight_style,
            diagnostic_severity: chunk.diagnostic_severity,
            is_unnecessary: chunk.is_unnecessary,
            is_tab: chunk.is_tab,
            is_inlay: chunk.is_inlay,
            underline: chunk.underline,
            renderer: chunk.renderer,
        })
    }
}

impl ConcealRows<'_> {
    pub(crate) fn seek(&mut self, row: u32) {
        let conceal_point = ConcealPoint::new(row, 0);
        self.cursor.seek(&conceal_point, Bias::Left);
        let overshoot = conceal_point.0 - self.cursor.start().0.0;
        let fold_point = FoldPoint(self.cursor.start().1.0 + overshoot);
        self.input_rows.seek(fold_point.row());
        self.conceal_point = conceal_point;
    }
}

impl Iterator for ConcealRows<'_> {
    type Item = RowInfo;

    fn next(&mut self) -> Option<Self::Item> {
        let mut traversed_concealment = false;
        while self.conceal_point > self.cursor.end().0 {
            self.cursor.next();
            traversed_concealment = true;
            if self.cursor.item().is_none() {
                break;
            }
        }

        if self.cursor.item().is_some() {
            if traversed_concealment {
                self.input_rows.seek(self.cursor.start().1.0.row);
                self.input_rows.next();
            }
            *self.conceal_point.row_mut() += 1;
            self.input_rows.next()
        } else {
            None
        }
    }
}

// --- Helper functions ---

/// Builds the SumTree<Transform> from scratch by resolving buffer-anchor
/// concealments into fold-offset space, then emitting alternating isomorphic
/// and replacement nodes.
///
/// The resolution pipeline for each concealment:
///   buffer Anchor → buffer offset → inlay point → fold point → fold offset
///
/// Concealments that overlap a revealed range or collapse to zero width
/// (e.g. inside a fold) are skipped.
fn build_transforms(
    transforms: &mut SumTree<Transform>,
    fold_snapshot: &FoldSnapshot,
    concealments: &[(Range<Anchor>, SharedString)],
    revealed_ranges: &[Range<Anchor>],
) {
    let buffer = &fold_snapshot.inlay_snapshot.buffer;

    // Phase 1: resolve each concealment's buffer anchors to fold offsets,
    // filtering out revealed and degenerate ones.
    let mut resolved: Vec<(Range<FoldOffset>, SharedString)> = concealments
        .iter()
        .filter_map(|(range, replacement)| {
            // Skip concealments that overlap with any revealed range.
            // This is what makes line-based cursor reveal work: the cursor's
            // line is added to revealed_ranges, suppressing concealments there.
            if revealed_ranges.iter().any(|revealed| {
                range.start.cmp(&revealed.end, buffer).is_lt()
                    && range.end.cmp(&revealed.start, buffer).is_gt()
            }) {
                return None;
            }

            // Walk through the coordinate layers: buffer → inlay → fold
            let start_buffer_offset = range.start.to_offset(buffer);
            let end_buffer_offset = range.end.to_offset(buffer);

            let start_inlay_point = fold_snapshot
                .inlay_snapshot
                .to_inlay_point(buffer.offset_to_point(start_buffer_offset));
            let end_inlay_point = fold_snapshot
                .inlay_snapshot
                .to_inlay_point(buffer.offset_to_point(end_buffer_offset));

            let start_fold_point = fold_snapshot.to_fold_point(start_inlay_point, Bias::Right);
            let end_fold_point = fold_snapshot.to_fold_point(end_inlay_point, Bias::Left);

            let start_fold = start_fold_point.to_offset(fold_snapshot);
            let end_fold = end_fold_point.to_offset(fold_snapshot);

            // Zero-width after folding means the concealment is entirely inside
            // a fold — skip it since it's already hidden.
            if start_fold >= end_fold {
                return None;
            }

            Some((start_fold..end_fold, replacement.clone()))
        })
        .collect();

    // Phase 2: sort and deduplicate overlapping concealments (keep the first).
    resolved.sort_by_key(|(range, _)| range.start);
    resolved.dedup_by(|b, a| b.0.start < a.0.end);

    // Phase 3: build the tree by walking sorted concealments left to right,
    // emitting isomorphic gaps between them and replacement nodes for each.
    let mut offset = FoldOffset(MultiBufferOffset(0));
    for (range, replacement) in &resolved {
        // Emit isomorphic node for the gap before this concealment.
        if range.start > offset {
            let text_summary = fold_snapshot.text_summary_for_range(
                offset.to_point(fold_snapshot)..range.start.to_point(fold_snapshot),
            );
            push_isomorphic(transforms, text_summary);
        }

        // Emit a replacement node: input is the original text summary,
        // output is the replacement string's summary. This is the core of
        // concealment — the display pipeline sees the shorter replacement
        // while the buffer retains the original text.
        let input_summary = fold_snapshot.text_summary_for_range(
            range.start.to_point(fold_snapshot)..range.end.to_point(fold_snapshot),
        );

        transforms.push(
            Transform::Replacement {
                input: input_summary,
                text: replacement.clone(),
            },
            (),
        );

        offset = range.end;
    }

    let total = FoldOffset(fold_snapshot.text_summary().len);
    if offset < total {
        let text_summary = fold_snapshot
            .text_summary_for_range(offset.to_point(fold_snapshot)..total.to_point(fold_snapshot));
        push_isomorphic(transforms, text_summary);
    }

    if transforms.is_empty() {
        let text_summary = fold_snapshot.text_summary();
        push_isomorphic(transforms, text_summary);
    }
}

/// Pushes an isomorphic (passthrough) transform, merging with the previous node
/// if it's also isomorphic. This keeps the tree compact — consecutive passthrough
/// regions become a single node rather than many small ones.
fn push_isomorphic(transforms: &mut SumTree<Transform>, summary: MBTextSummary) {
    let mut did_merge = false;
    transforms.update_last(
        |last| {
            if let Transform::Isomorphic(existing) = last {
                *existing += summary;
                did_merge = true;
            }
        },
        (),
    );
    if !did_merge {
        transforms.push(Transform::Isomorphic(summary), ());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultiBuffer, display_map::inlay_map::InlayMap};
    use gpui;
    use multi_buffer::MultiBufferOffset as O;
    use settings::SettingsStore;

    fn init_test(cx: &mut gpui::App) {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
    }

    #[gpui::test]
    fn test_conceal_then_reveal(cx: &mut gpui::App) {
        init_test(cx);
        let text = "x = 2\nlambda x: x + 1\n";
        let buffer = MultiBuffer::build_simple(text, cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot.clone());

        let concealments = vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(12)),
            SharedString::from("λ"),
        )];
        let (snapshot, _) = conceal_map.set_concealments(concealments);
        assert_eq!(snapshot.text(), "x = 2\nλ x: x + 1\n");

        // Reveal via the deferred path (same as the real editor code path).
        // set_revealed_ranges marks dirty, then read() triggers sync().
        let revealed =
            vec![buffer_snapshot.anchor_before(O(6))..buffer_snapshot.anchor_after(O(12))];
        conceal_map.set_revealed_ranges(revealed);
        let (snapshot, _) = conceal_map.read(fold_snapshot.clone(), vec![]);
        assert_eq!(snapshot.text(), "x = 2\nlambda x: x + 1\n");

        // Re-conceal (cursor moved away)
        conceal_map.set_revealed_ranges(vec![]);
        let (snapshot, _) = conceal_map.read(fold_snapshot, vec![]);
        assert_eq!(snapshot.text(), "x = 2\nλ x: x + 1\n");
    }

    #[gpui::test]
    fn test_basic_concealment(cx: &mut gpui::App) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple("hello != world", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, snapshot) = ConcealMap::new(fold_snapshot);

        assert_eq!(snapshot.text(), "hello != world");

        let start = buffer_snapshot.anchor_after(O(6));
        let end = buffer_snapshot.anchor_before(O(8));

        let (snapshot, _edits) = conceal_map.set_concealments(vec![(start..end, "≠".into())]);

        assert_eq!(snapshot.text(), "hello ≠ world");
    }

    #[gpui::test]
    fn test_multiple_concealments(cx: &mut gpui::App) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple("a != b && c", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        let (snapshot, _) = conceal_map.set_concealments(vec![
            (
                buffer_snapshot.anchor_after(O(2))..buffer_snapshot.anchor_before(O(4)),
                "≠".into(),
            ),
            (
                buffer_snapshot.anchor_after(O(7))..buffer_snapshot.anchor_before(O(9)),
                "∧".into(),
            ),
        ]);

        assert_eq!(snapshot.text(), "a ≠ b ∧ c");
    }

    #[gpui::test]
    fn test_concealment_point_conversion(cx: &mut gpui::App) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple("lambda x: x", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(6)),
            "λ".into(),
        )]);

        // "lambda x: x" -> "λ x: x"
        assert_eq!(snapshot.text(), "λ x: x");

        // The "x" after "λ " is at conceal column 4 (λ=2bytes, space=1, x=1 => offset 4)
        // But in fold space it's at column 7 (lambda=6, space=1)
        let fold_point_for_x = FoldPoint::new(0, 7);
        let conceal_point = snapshot.to_conceal_point(fold_point_for_x, Bias::Right);
        // "λ" is 2 bytes as UTF-8, so conceal column for the space after λ is 2
        // then "x" at column 3
        assert_eq!(conceal_point, ConcealPoint::new(0, 3));

        let back = conceal_point.to_fold_point(&snapshot);
        assert_eq!(back, fold_point_for_x);
    }

    #[gpui::test]
    fn test_concealment_clear(cx: &mut gpui::App) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple("hello != world", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "hello ≠ world");

        let (snapshot, _) = conceal_map.set_concealments(vec![]);
        assert_eq!(snapshot.text(), "hello != world");
    }

    #[gpui::test]
    fn test_concealment_with_buffer_edit(cx: &mut gpui::App) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple("a != b", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (mut inlay_map, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (mut fold_map, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(2))..buffer_snapshot.anchor_before(O(4)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "a ≠ b");

        let subscription = buffer.update(cx, |buffer, _| buffer.subscribe());

        // Edit the buffer: "a != b" -> "a != bcd"
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(O(6)..O(6), "cd")], None, cx);
        });
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let edits = subscription.consume().into_inner();

        let (inlay_snapshot, inlay_edits) = inlay_map.sync(buffer_snapshot, edits);
        let (fold_snapshot, fold_edits) = fold_map.read(inlay_snapshot, inlay_edits);
        let (snapshot, _conceal_edits) = conceal_map.read(fold_snapshot, fold_edits);

        assert_eq!(snapshot.text(), "a ≠ bcd");
    }
}
