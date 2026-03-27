// Display pipeline layer for visual text replacement.
//   InlayMap → FoldMap → **ConcealMap** → TabMap → WrapMap → BlockMap

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
use text::Patch;

pub struct ConcealMap {
    snapshot: ConcealSnapshot,
    concealments: Vec<(Range<Anchor>, SharedString)>,
    revealed_ranges: Vec<Range<Anchor>>,
    prev_revealed_ranges: Vec<Range<Anchor>>,
}

#[derive(Clone)]
pub struct ConcealSnapshot {
    pub fold_snapshot: FoldSnapshot,
    transforms: SumTree<Transform>,
    /// When true, fold space == conceal space — all coordinate conversions
    /// and chunk iteration skip the SumTree entirely.
    passthrough: bool,
    pub version: usize,
}

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
    passthrough: bool,
    transform_cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealOffset, FoldOffset>>,
    fold_chunks: FoldChunks<'a>,
    fold_chunk: Option<(FoldOffset, Chunk<'a>)>,
    fold_offset: FoldOffset,
    output_offset: ConcealOffset,
    max_output_offset: ConcealOffset,
    replacement_offset: usize,
    /// True after a concealment — the next isomorphic region needs to seek
    /// fold_chunks to the right position. False during sequential access
    /// where fold_chunks.next() suffices.
    needs_seek: bool,
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
        if snapshot.passthrough {
            return FoldPoint(self.0);
        }
        let (start, _, _) = snapshot
            .transforms
            .find::<Dimensions<ConcealPoint, FoldPoint>, _>((), &self, Bias::Right);
        let overshoot = self.0 - start.0.0;
        FoldPoint(start.1.0 + overshoot)
    }

    pub fn to_offset(self, snapshot: &ConcealSnapshot) -> ConcealOffset {
        if snapshot.passthrough {
            return ConcealOffset(FoldPoint(self.0).to_offset(&snapshot.fold_snapshot).0);
        }
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
                debug_assert!(!transform.is_concealment());
                let end_fold_offset =
                    FoldPoint(start.1.input.lines + overshoot).to_offset(&snapshot.fold_snapshot);
                offset += end_fold_offset.0 - start.1.input.len;
            } else {
                return snapshot.len();
            }
        }
        ConcealOffset(offset)
    }
}

impl ConcealOffset {
    pub fn to_point(self, snapshot: &ConcealSnapshot) -> ConcealPoint {
        if snapshot.passthrough {
            return ConcealPoint(FoldOffset(self.0).to_point(&snapshot.fold_snapshot).0);
        }
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

impl ConcealMap {
    pub fn new(fold_snapshot: FoldSnapshot) -> (Self, ConcealSnapshot) {
        let mut snapshot = ConcealSnapshot {
            transforms: SumTree::default(),
            fold_snapshot,
            passthrough: true,
            version: 0,
        };
        build_transforms(&mut snapshot.transforms, &snapshot.fold_snapshot, &[], &[]);
        (
            Self {
                snapshot: snapshot.clone(),
                concealments: Vec::new(),
                revealed_ranges: Vec::new(),
                prev_revealed_ranges: Vec::new(),
            },
            snapshot,
        )
    }

    pub fn read(
        &mut self,
        fold_snapshot: FoldSnapshot,
        fold_edits: Vec<FoldEdit>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let edits = self.sync(fold_snapshot, fold_edits);
        (self.snapshot.clone(), edits)
    }

    /// Absorb a new fold snapshot without rebuilding transforms. Use before
    /// `set_concealments` to avoid a redundant rebuild.
    pub fn sync_fold_snapshot(&mut self, fold_snapshot: FoldSnapshot) {
        self.snapshot.fold_snapshot = fold_snapshot;
        self.prev_revealed_ranges = self.revealed_ranges.clone();
    }

    pub fn set_concealments(
        &mut self,
        concealments: Vec<(Range<Anchor>, SharedString)>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let old_snapshot = self.snapshot.clone();
        self.snapshot.passthrough = concealments.is_empty();
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

    pub fn set_revealed_ranges(&mut self, revealed_ranges: Vec<Range<Anchor>>) {
        std::mem::swap(&mut self.prev_revealed_ranges, &mut self.revealed_ranges);
        self.revealed_ranges = revealed_ranges;
    }

    /// Reconciles the conceal layer with upstream fold changes. Both reveal
    /// changes and fold edits are handled incrementally by splicing affected
    /// regions in the existing transform tree.
    fn sync(&mut self, fold_snapshot: FoldSnapshot, fold_edits: Vec<FoldEdit>) -> Vec<ConcealEdit> {
        let reveal_changed = self.revealed_ranges != self.prev_revealed_ranges;

        if fold_edits.is_empty()
            && self.snapshot.fold_snapshot.version == fold_snapshot.version
            && !reveal_changed
        {
            return Vec::new();
        }

        self.snapshot.fold_snapshot = fold_snapshot;
        self.snapshot.version += 1;

        // Reveal-only change: compute the affected fold-offset ranges from
        // the symmetric difference of old and new revealed ranges, then splice
        // just those regions instead of rebuilding the entire tree.
        if fold_edits.is_empty() {
            let affected = self.reveal_affected_ranges();
            if affected.is_empty() {
                self.prev_revealed_ranges = self.revealed_ranges.clone();
                return vec![];
            }

            let resolved = self.resolve_concealments();
            let mut conceal_edits = Patch::default();
            let mut new_transforms = SumTree::default();
            let mut cursor = self
                .snapshot
                .transforms
                .cursor::<Dimensions<FoldOffset, ConcealOffset>>(());

            for affected_range in &affected {
                new_transforms.append(cursor.slice(&affected_range.start, Bias::Left), ());
                if cursor.item().is_some_and(|t| !t.is_concealment())
                    && cursor.end().0 == affected_range.start
                {
                    if let Some(Transform::Isomorphic(summary)) = cursor.item() {
                        push_isomorphic(&mut new_transforms, *summary);
                    }
                    cursor.next();
                }

                let old_start =
                    cursor.start().1 + (affected_range.start.0 - cursor.start().0.0);
                cursor.seek(&affected_range.end, Bias::Right);
                let old_end =
                    cursor.start().1 + (affected_range.end.0 - cursor.start().0.0);

                let prefix_start = FoldOffset(new_transforms.summary().input.len);
                if affected_range.start > prefix_start {
                    push_isomorphic(
                        &mut new_transforms,
                        self.snapshot.fold_snapshot.text_summary_for_range(
                            prefix_start.to_point(&self.snapshot.fold_snapshot)
                                ..affected_range.start.to_point(&self.snapshot.fold_snapshot),
                        ),
                    );
                }
                let new_start = ConcealOffset(new_transforms.summary().output.len);

                self.build_region(
                    &mut new_transforms,
                    &resolved,
                    affected_range.start,
                    affected_range.end,
                );

                let built = FoldOffset(new_transforms.summary().input.len);
                if built < affected_range.end {
                    push_isomorphic(
                        &mut new_transforms,
                        self.snapshot.fold_snapshot.text_summary_for_range(
                            built.to_point(&self.snapshot.fold_snapshot)
                                ..affected_range.end.to_point(&self.snapshot.fold_snapshot),
                        ),
                    );
                }
                let new_end = ConcealOffset(new_transforms.summary().output.len);

                conceal_edits.push(text::Edit {
                    old: old_start..old_end,
                    new: new_start..new_end,
                });

                if cursor.item().is_some_and(|t| !t.is_concealment())
                    && cursor.start().0 < cursor.end().0
                {
                    let remainder_start = FoldOffset(new_transforms.summary().input.len);
                    let remainder_end = cursor.end().0;
                    if remainder_end > remainder_start {
                        push_isomorphic(
                            &mut new_transforms,
                            self.snapshot.fold_snapshot.text_summary_for_range(
                                remainder_start.to_point(&self.snapshot.fold_snapshot)
                                    ..remainder_end.to_point(&self.snapshot.fold_snapshot),
                            ),
                        );
                    }
                    cursor.next();
                }
            }

            new_transforms.append(cursor.suffix(), ());
            if new_transforms.is_empty() {
                push_isomorphic(
                    &mut new_transforms,
                    self.snapshot.fold_snapshot.text_summary(),
                );
            }

            drop(cursor);
            self.snapshot.transforms = new_transforms;
            self.prev_revealed_ranges = self.revealed_ranges.clone();
            return conceal_edits.into_inner();
        }

        // Incremental sync: walk the old transform tree and splice in edits,
        // resolving concealments only in the edited regions. Follows the
        // same pattern as InlayMap::sync.
        let resolved = self.resolve_concealments();
        let mut conceal_edits = Patch::default();
        let mut new_transforms = SumTree::default();
        let mut cursor = self
            .snapshot
            .transforms
            .cursor::<Dimensions<FoldOffset, ConcealOffset>>(());
        let mut fold_edits_iter = fold_edits.iter().peekable();

        while let Some(fold_edit) = fold_edits_iter.next() {
            new_transforms.append(cursor.slice(&fold_edit.old.start, Bias::Left), ());
            if cursor.item().is_some_and(|t| !t.is_concealment())
                && cursor.end().0 == fold_edit.old.start
            {
                if let Some(Transform::Isomorphic(summary)) = cursor.item() {
                    push_isomorphic(&mut new_transforms, *summary);
                }
                cursor.next();
            }

            let old_start = cursor.start().1 + (fold_edit.old.start.0 - cursor.start().0.0);
            cursor.seek(&fold_edit.old.end, Bias::Right);
            let old_end = cursor.start().1 + (fold_edit.old.end.0 - cursor.start().0.0);

            // Build the new content for the edited region: isomorphic text
            // interspersed with any concealments that fall in range.
            let prefix_start = FoldOffset(new_transforms.summary().input.len);
            let prefix_end = fold_edit.new.start;
            if prefix_end > prefix_start {
                push_isomorphic(
                    &mut new_transforms,
                    self.snapshot.fold_snapshot.text_summary_for_range(
                        prefix_start.to_point(&self.snapshot.fold_snapshot)
                            ..prefix_end.to_point(&self.snapshot.fold_snapshot),
                    ),
                );
            }
            let new_start = ConcealOffset(new_transforms.summary().output.len);

            self.build_region(
                &mut new_transforms,
                &resolved,
                fold_edit.new.start,
                fold_edit.new.end,
            );

            let built = FoldOffset(new_transforms.summary().input.len);
            if built < fold_edit.new.end {
                push_isomorphic(
                    &mut new_transforms,
                    self.snapshot.fold_snapshot.text_summary_for_range(
                        built.to_point(&self.snapshot.fold_snapshot)
                            ..fold_edit.new.end.to_point(&self.snapshot.fold_snapshot),
                    ),
                );
            }
            let new_end = ConcealOffset(new_transforms.summary().output.len);

            conceal_edits.push(text::Edit {
                old: old_start..old_end,
                new: new_start..new_end,
            });

            // If the next edit doesn't intersect the current transform,
            // push its remainder.
            if fold_edits_iter
                .peek()
                .is_none_or(|edit| edit.old.start >= cursor.end().0)
            {
                let remainder_start = FoldOffset(new_transforms.summary().input.len);
                let remainder_end =
                    FoldOffset(fold_edit.new.end.0 + (cursor.end().0.0 - fold_edit.old.end.0));
                if remainder_end > remainder_start {
                    self.build_region(
                        &mut new_transforms,
                        &resolved,
                        remainder_start,
                        remainder_end,
                    );
                    let built = FoldOffset(new_transforms.summary().input.len);
                    if built < remainder_end {
                        push_isomorphic(
                            &mut new_transforms,
                            self.snapshot.fold_snapshot.text_summary_for_range(
                                built.to_point(&self.snapshot.fold_snapshot)
                                    ..remainder_end.to_point(&self.snapshot.fold_snapshot),
                            ),
                        );
                    }
                }
                cursor.next();
            }
        }

        new_transforms.append(cursor.suffix(), ());
        if new_transforms.is_empty() {
            push_isomorphic(
                &mut new_transforms,
                self.snapshot.fold_snapshot.text_summary(),
            );
        }

        drop(cursor);
        self.snapshot.transforms = new_transforms;
        self.prev_revealed_ranges = self.revealed_ranges.clone();
        conceal_edits.into_inner()
    }

    fn resolve_concealments(&self) -> Vec<(Range<FoldOffset>, SharedString)> {
        resolve_concealments(
            &self.snapshot.fold_snapshot,
            &self.concealments,
            &self.revealed_ranges,
        )
    }

    /// Returns fold-offset ranges of concealments whose reveal state changed
    /// between prev_revealed_ranges and revealed_ranges.
    fn reveal_affected_ranges(&self) -> Vec<Range<FoldOffset>> {
        if self.concealments.is_empty() {
            return Vec::new();
        }
        let buffer = &self.snapshot.fold_snapshot.inlay_snapshot.buffer;
        let fold_snapshot = &self.snapshot.fold_snapshot;

        let mut affected = Vec::new();
        for (range, _) in &self.concealments {
            let was_revealed = self.prev_revealed_ranges.iter().any(|revealed| {
                range.start.cmp(&revealed.end, buffer).is_lt()
                    && range.end.cmp(&revealed.start, buffer).is_gt()
            });
            let is_revealed = self.revealed_ranges.iter().any(|revealed| {
                range.start.cmp(&revealed.end, buffer).is_lt()
                    && range.end.cmp(&revealed.start, buffer).is_gt()
            });
            if was_revealed == is_revealed {
                continue;
            }
            let start_offset = range.start.to_offset(buffer);
            let end_offset = range.end.to_offset(buffer);
            let start_fold = fold_snapshot
                .to_fold_point(
                    fold_snapshot
                        .inlay_snapshot
                        .to_inlay_point(buffer.offset_to_point(start_offset)),
                    Bias::Right,
                )
                .to_offset(fold_snapshot);
            let end_fold = fold_snapshot
                .to_fold_point(
                    fold_snapshot
                        .inlay_snapshot
                        .to_inlay_point(buffer.offset_to_point(end_offset)),
                    Bias::Left,
                )
                .to_offset(fold_snapshot);
            if start_fold < end_fold {
                affected.push(start_fold..end_fold);
            }
        }
        affected.sort_by_key(|r| r.start);
        affected
    }

    fn build_region(
        &self,
        transforms: &mut SumTree<Transform>,
        resolved: &[(Range<FoldOffset>, SharedString)],
        start: FoldOffset,
        end: FoldOffset,
    ) {
        let fold_snapshot = &self.snapshot.fold_snapshot;
        for (range, replacement) in resolved {
            if range.end <= start || range.start >= end {
                continue;
            }
            let clamped_start = range.start.max(start);
            let clamped_end = range.end.min(end);

            let built = FoldOffset(transforms.summary().input.len);
            if clamped_start > built {
                push_isomorphic(
                    transforms,
                    fold_snapshot.text_summary_for_range(
                        built.to_point(fold_snapshot)..clamped_start.to_point(fold_snapshot),
                    ),
                );
            }

            let input_summary = fold_snapshot.text_summary_for_range(
                clamped_start.to_point(fold_snapshot)..clamped_end.to_point(fold_snapshot),
            );
            transforms.push(
                Transform::Replacement {
                    input: input_summary,
                    text: replacement.clone(),
                },
                (),
            );
        }
    }
}

// --- ConcealSnapshot impl ---

impl ConcealSnapshot {
    /// Maps a fold-space point to conceal-space. If the point falls inside a
    /// concealment, it snaps to the start or end depending on bias (Left snaps
    /// to start, Right snaps to end — this controls cursor behavior at boundaries).
    pub fn to_conceal_point(&self, point: FoldPoint, bias: Bias) -> ConcealPoint {
        if self.passthrough {
            return ConcealPoint(point.0);
        }
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
        if self.passthrough {
            return ConcealOffset(self.fold_snapshot.len().0);
        }
        ConcealOffset(self.transforms.summary().output.len)
    }

    pub fn max_point(&self) -> ConcealPoint {
        if self.passthrough {
            return ConcealPoint(self.fold_snapshot.max_point().0);
        }
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
        if self.passthrough {
            return ConcealPoint(self.fold_snapshot.clip_point(FoldPoint(point.0), bias).0);
        }
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
        ConcealPointCursor {
            passthrough: self.passthrough,
            cursor,
        }
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
        if self.passthrough {
            let fold_start = FoldOffset(range.start.0);
            let fold_end = FoldOffset(range.end.0);
            return ConcealChunks {
                passthrough: true,
                transform_cursor: self
                    .transforms
                    .cursor::<Dimensions<ConcealOffset, FoldOffset>>(()),
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
                needs_seek: false,
            };
        }

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
            passthrough: false,
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
            needs_seek: false,
        }
    }

    pub fn text_summary_for_range(&self, range: Range<ConcealPoint>) -> MBTextSummary {
        if self.passthrough {
            let fold_start = FoldPoint(range.start.0);
            let fold_end = FoldPoint(range.end.0);
            return self
                .fold_snapshot
                .text_summary_for_range(fold_start..fold_end);
        }

        // Walk the transform tree to accumulate the correct output-space summary.
        // Delegating to fold_snapshot would return fold-space (input) lengths,
        // which are wrong when concealments shrink the displayed text.
        let start_offset = range.start.to_offset(self);
        let end_offset = range.end.to_offset(self);
        let mut summary = MBTextSummary::default();
        for chunk in self.chunks(start_offset..end_offset, false, Highlights::default()) {
            summary += MBTextSummary::from(chunk.text);
        }
        summary
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
    passthrough: bool,
    cursor: Cursor<'transforms, 'static, Transform, Dimensions<FoldPoint, ConcealPoint>>,
}

impl ConcealPointCursor<'_> {
    pub fn map(&mut self, point: FoldPoint, bias: Bias) -> ConcealPoint {
        if self.passthrough {
            return ConcealPoint(point.0);
        }
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
        if self.passthrough {
            self.fold_chunks
                .seek(FoldOffset(range.start.0)..FoldOffset(range.end.0));
            self.output_offset = range.start;
            self.max_output_offset = range.end;
            return;
        }

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
        self.needs_seek = false;
    }
}

impl<'a> Iterator for ConcealChunks<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.passthrough {
            return self.fold_chunks.next();
        }

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
            self.needs_seek = true;
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
        if self.fold_chunk.is_none() {
            if self.needs_seek {
                let transform_end = self.transform_cursor.end();
                let fold_end = if self.max_output_offset < transform_end.0 {
                    let overshoot = self.max_output_offset - self.transform_cursor.start().0;
                    self.transform_cursor.start().1 + overshoot
                } else {
                    transform_end.1
                };
                self.fold_chunks.seek(self.fold_offset..fold_end);
                self.needs_seek = false;
            }
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
/// Resolves buffer-anchor concealments to fold-offset ranges, filtering out
/// revealed ranges and zero-width results. Sorts and deduplicates overlaps.
fn resolve_concealments(
    fold_snapshot: &FoldSnapshot,
    concealments: &[(Range<Anchor>, SharedString)],
    revealed_ranges: &[Range<Anchor>],
) -> Vec<(Range<FoldOffset>, SharedString)> {
    let buffer = &fold_snapshot.inlay_snapshot.buffer;

    let mut resolved: Vec<(Range<FoldOffset>, SharedString)> = concealments
        .iter()
        .filter_map(|(range, replacement)| {
            if revealed_ranges.iter().any(|revealed| {
                range.start.cmp(&revealed.end, buffer).is_lt()
                    && range.end.cmp(&revealed.start, buffer).is_gt()
            }) {
                return None;
            }

            let start_buffer_offset = range.start.to_offset(buffer);
            let end_buffer_offset = range.end.to_offset(buffer);
            let start_inlay_point = fold_snapshot
                .inlay_snapshot
                .to_inlay_point(buffer.offset_to_point(start_buffer_offset));
            let end_inlay_point = fold_snapshot
                .inlay_snapshot
                .to_inlay_point(buffer.offset_to_point(end_buffer_offset));
            let start_fold = fold_snapshot
                .to_fold_point(start_inlay_point, Bias::Right)
                .to_offset(fold_snapshot);
            let end_fold = fold_snapshot
                .to_fold_point(end_inlay_point, Bias::Left)
                .to_offset(fold_snapshot);

            if start_fold >= end_fold {
                return None;
            }
            Some((start_fold..end_fold, replacement.clone()))
        })
        .collect();

    resolved.sort_by_key(|(range, _)| range.start);
    let mut last_end = FoldOffset(MultiBufferOffset(0));
    resolved.retain(|(range, _)| {
        if range.start < last_end {
            return false;
        }
        last_end = range.end;
        true
    });
    resolved
}

fn build_transforms(
    transforms: &mut SumTree<Transform>,
    fold_snapshot: &FoldSnapshot,
    concealments: &[(Range<Anchor>, SharedString)],
    revealed_ranges: &[Range<Anchor>],
) {
    let resolved = resolve_concealments(fold_snapshot, concealments, revealed_ranges);

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

    fn setup(
        text: &str,
        cx: &mut gpui::App,
    ) -> (
        ConcealMap,
        FoldSnapshot,
        multi_buffer::MultiBufferSnapshot,
    ) {
        init_test(cx);
        let buffer = MultiBuffer::build_simple(text, cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (conceal_map, _) = ConcealMap::new(fold_snapshot.clone());
        (conceal_map, fold_snapshot, buffer_snapshot)
    }

    #[gpui::test]
    fn test_conceal_then_reveal(cx: &mut gpui::App) {
        let (mut conceal_map, fold_snapshot, buffer_snapshot) =
            setup("x = 2\nlambda x: x + 1\n", cx);

        let concealments = vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(12)),
            SharedString::from("λ"),
        )];
        let (snapshot, _) = conceal_map.set_concealments(concealments);
        assert_eq!(snapshot.text(), "x = 2\nλ x: x + 1\n");

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
        let (mut conceal_map, _, buffer_snapshot) = setup("hello != world", cx);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
            "≠".into(),
        )]);

        assert_eq!(snapshot.text(), "hello ≠ world");
    }

    #[gpui::test]
    fn test_multiple_concealments(cx: &mut gpui::App) {
        let (mut conceal_map, _, buffer_snapshot) = setup("a != b && c", cx);

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
        let (mut conceal_map, _, buffer_snapshot) = setup("lambda x: x", cx);

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
        let (mut conceal_map, _, buffer_snapshot) = setup("hello != world", cx);

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

    #[gpui::test]
    fn test_text_summary_across_concealment(cx: &mut gpui::App) {
        let (mut conceal_map, _, buffer_snapshot) = setup("lambda x", cx);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(6)),
            "λ".into(),
        )]);
        assert_eq!(snapshot.text(), "λ x");

        // Full range summary should match the concealed text, not the original.
        let summary =
            snapshot.text_summary_for_range(ConcealPoint::new(0, 0)..snapshot.max_point());
        assert_eq!(summary.len, O(4)); // "λ x" = 2 + 1 + 1 = 4 bytes
        assert_eq!(summary.lines, Point::new(0, 4));
    }

    #[gpui::test]
    fn test_overlapping_concealments_keep_first(cx: &mut gpui::App) {
        let (mut conceal_map, _, buffer_snapshot) = setup("a !== b", cx);

        let (snapshot, _) = conceal_map.set_concealments(vec![
            (
                buffer_snapshot.anchor_after(O(2))..buffer_snapshot.anchor_before(O(4)),
                "≠".into(),
            ),
            (
                buffer_snapshot.anchor_after(O(3))..buffer_snapshot.anchor_before(O(5)),
                "≡".into(),
            ),
        ]);

        // "!=" at 2..4 wins, "==" at 3..5 overlaps and is dropped.
        assert_eq!(snapshot.text(), "a ≠= b");
    }

    #[gpui::test]
    fn test_incremental_sync_edit_near_concealment(cx: &mut gpui::App) {
        init_test(cx);
        // Edit text before a concealment — verify the concealment survives
        // and the sync produces correct output.
        let buffer = MultiBuffer::build_simple("hello != world", cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (mut inlay_map, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (mut fold_map, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "hello ≠ world");

        let subscription = buffer.update(cx, |buffer, _| buffer.subscribe());

        // Insert "dear " before "hello" — shifts the concealment right.
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(O(0)..O(0), "dear ")], None, cx);
        });
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let edits = subscription.consume().into_inner();

        let (inlay_snapshot, inlay_edits) = inlay_map.sync(buffer_snapshot, edits);
        let (fold_snapshot, fold_edits) = fold_map.read(inlay_snapshot, inlay_edits);
        let (snapshot, _) = conceal_map.read(fold_snapshot, fold_edits);

        assert_eq!(snapshot.text(), "dear hello ≠ world");
    }

    #[gpui::test]
    fn test_incremental_sync_edit_after_concealment(cx: &mut gpui::App) {
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

        // Append " and c" after "b".
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(O(6)..O(6), " and c")], None, cx);
        });
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let edits = subscription.consume().into_inner();

        let (inlay_snapshot, inlay_edits) = inlay_map.sync(buffer_snapshot, edits);
        let (fold_snapshot, fold_edits) = fold_map.read(inlay_snapshot, inlay_edits);
        let (snapshot, _) = conceal_map.read(fold_snapshot, fold_edits);

        assert_eq!(snapshot.text(), "a ≠ b and c");
    }

    #[gpui::test]
    fn test_multi_line_concealment(cx: &mut gpui::App) {
        let (mut conceal_map, _, buffer_snapshot) = setup("if (\n  true\n):\n  pass\n", cx);

        // Conceal "if" (offsets 0..2) with a single character; the space remains.
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(2)),
            "⌥".into(),
        )]);
        assert_eq!(snapshot.text(), "⌥ (\n  true\n):\n  pass\n");

        // Verify max_point and len reflect the concealed output.
        assert_eq!(snapshot.max_point(), ConcealPoint::new(4, 0));
        assert_eq!(snapshot.len().0, O("⌥ (\n  true\n):\n  pass\n".len()));
    }
}
