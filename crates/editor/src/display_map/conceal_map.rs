// Display pipeline layer for visual text replacement.
//   InlayMap → FoldMap → **ConcealMap** → TabMap → WrapMap → BlockMap

use super::{
    Highlights,
    fold_map::{Chunk, FoldChunks, FoldEdit, FoldOffset, FoldPoint, FoldRows, FoldSnapshot},
};
use collections::HashSet;
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
    /// Cached fold-space ranges for each entry in `concealments`.
    /// Avoids re-converting anchors → fold offsets on every reveal change.
    /// Recomputed when the fold snapshot or concealments change.
    cached_fold_ranges: Vec<Option<Range<FoldOffset>>>,
    /// Indices into `concealments` that are currently revealed (cursor is on them).
    revealed_indices: HashSet<usize>,
    prev_revealed_indices: HashSet<usize>,
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

impl<'a> sum_tree::Dimension<'a, TransformSummary> for ConcealPoint {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += &summary.output.lines;
    }
}

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

pub struct ConcealChunks<'a> {
    passthrough: bool,
    transforms: &'a SumTree<Transform>,
    transform_cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealOffset, FoldOffset>>,
    fold_chunks: FoldChunks<'a>,
    fold_chunk: Option<(FoldOffset, Chunk<'a>)>,
    fold_offset: FoldOffset,
    output_offset: ConcealOffset,
    max_output_offset: ConcealOffset,
    max_fold_offset: FoldOffset,
}

#[derive(Clone)]
pub struct ConcealRows<'a> {
    cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealPoint, FoldPoint>>,
    input_rows: FoldRows<'a>,
    conceal_point: ConcealPoint,
}

impl ConcealPoint {
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
        push_isomorphic(
            &mut snapshot.transforms,
            snapshot.fold_snapshot.text_summary(),
        );
        (
            Self {
                snapshot: snapshot.clone(),
                concealments: Vec::new(),
                cached_fold_ranges: Vec::new(),
                revealed_indices: HashSet::default(),
                prev_revealed_indices: HashSet::default(),
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

    pub fn sync_fold_snapshot(&mut self, fold_snapshot: FoldSnapshot) {
        self.snapshot.fold_snapshot = fold_snapshot;
        self.prev_revealed_indices = self.revealed_indices.clone();
    }

    #[ztracing::instrument(skip_all)]
    pub fn set_concealments(
        &mut self,
        concealments: Vec<(Range<Anchor>, SharedString)>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let old_snapshot = self.snapshot.clone();
        self.snapshot.passthrough = concealments.is_empty();
        self.concealments = concealments;
        self.snapshot.version += 1;

        self.recompute_cached_fold_ranges();
        let resolved = self.resolve_concealments();

        let mut new_transforms = SumTree::default();
        build_transforms_from_resolved(
            &mut new_transforms,
            &self.snapshot.fold_snapshot,
            &resolved,
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

    pub fn set_revealed_indices(&mut self, indices: HashSet<usize>) {
        std::mem::swap(&mut self.prev_revealed_indices, &mut self.revealed_indices);
        self.revealed_indices = indices;
    }

    #[ztracing::instrument(skip_all)]
    fn sync(&mut self, fold_snapshot: FoldSnapshot, fold_edits: Vec<FoldEdit>) -> Vec<ConcealEdit> {
        let reveal_changed = self.revealed_indices != self.prev_revealed_indices;

        if fold_edits.is_empty()
            && self.snapshot.fold_snapshot.version == fold_snapshot.version
            && !reveal_changed
        {
            return Vec::new();
        }

        self.snapshot.fold_snapshot = fold_snapshot;
        self.snapshot.version += 1;

        if fold_edits.is_empty() {
            let affected = self.reveal_affected_ranges();
            if affected.is_empty() {
                self.prev_revealed_indices = self.revealed_indices.clone();
                return vec![];
            }

            let resolved = self.resolve_concealments();
            let mut conceal_edits = Patch::default();
            let mut new_transforms = SumTree::default();
            let mut cursor = self
                .snapshot
                .transforms
                .cursor::<Dimensions<FoldOffset, ConcealOffset>>(());

            let mut affected_iter = affected.iter().peekable();
            while let Some(affected_range) = affected_iter.next() {
                new_transforms.append(cursor.slice(&affected_range.start, Bias::Left), ());
                if cursor.item().is_some_and(|t| !t.is_concealment())
                    && cursor.end().0 == affected_range.start
                {
                    if let Some(Transform::Isomorphic(summary)) = cursor.item() {
                        push_isomorphic(&mut new_transforms, *summary);
                    }
                    cursor.next();
                }

                let old_start = cursor.start().1 + (affected_range.start.0 - cursor.start().0.0);
                cursor.seek(&affected_range.end, Bias::Right);
                let old_end = cursor.start().1 + (affected_range.end.0 - cursor.start().0.0);

                self.fill_isomorphic_gap(&mut new_transforms, affected_range.start);
                let new_start = ConcealOffset(new_transforms.summary().output.len);
                self.build_region_filled(
                    &mut new_transforms,
                    &resolved,
                    affected_range.start,
                    affected_range.end,
                );
                let new_end = ConcealOffset(new_transforms.summary().output.len);

                conceal_edits.push(text::Edit {
                    old: old_start..old_end,
                    new: new_start..new_end,
                });

                if cursor.item().is_some_and(|t| !t.is_concealment())
                    && cursor.start().0 < cursor.end().0
                    && affected_iter
                        .peek()
                        .is_none_or(|next| next.start >= cursor.end().0)
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
            self.prev_revealed_indices = self.revealed_indices.clone();
            return conceal_edits.into_inner();
        }

        self.recompute_cached_fold_ranges();
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

            self.fill_isomorphic_gap(&mut new_transforms, fold_edit.new.start);
            let new_start = ConcealOffset(new_transforms.summary().output.len);
            self.build_region_filled(
                &mut new_transforms,
                &resolved,
                fold_edit.new.start,
                fold_edit.new.end,
            );
            let new_end = ConcealOffset(new_transforms.summary().output.len);

            conceal_edits.push(text::Edit {
                old: old_start..old_end,
                new: new_start..new_end,
            });

            if fold_edits_iter
                .peek()
                .is_none_or(|edit| edit.old.start >= cursor.end().0)
            {
                let remainder_start = FoldOffset(new_transforms.summary().input.len);
                let remainder_end =
                    FoldOffset(fold_edit.new.end.0 + (cursor.end().0.0 - fold_edit.old.end.0));
                if remainder_end > remainder_start {
                    self.build_region_filled(
                        &mut new_transforms,
                        &resolved,
                        remainder_start,
                        remainder_end,
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
        self.prev_revealed_indices = self.revealed_indices.clone();
        conceal_edits.into_inner()
    }

    fn recompute_cached_fold_ranges(&mut self) {
        let fold_snapshot = &self.snapshot.fold_snapshot;
        let buffer = &fold_snapshot.inlay_snapshot.buffer;
        self.cached_fold_ranges = self
            .concealments
            .iter()
            .map(|(range, _)| {
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
                    Some(start_fold..end_fold)
                } else {
                    None
                }
            })
            .collect();
    }

    fn resolve_concealments(&self) -> Vec<(Range<FoldOffset>, SharedString)> {
        let mut resolved: Vec<_> = self
            .cached_fold_ranges
            .iter()
            .zip(&self.concealments)
            .enumerate()
            .filter_map(|(i, (fold_range, (_, replacement)))| {
                let fold_range = fold_range.as_ref()?;
                if self.revealed_indices.contains(&i) {
                    return None;
                }
                Some((fold_range.clone(), replacement.clone()))
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

    fn reveal_affected_ranges(&self) -> Vec<Range<FoldOffset>> {
        if self.concealments.is_empty() {
            return Vec::new();
        }

        let mut affected = Vec::new();
        for (i, fold_range) in self.cached_fold_ranges.iter().enumerate() {
            let Some(fold_range) = fold_range else {
                continue;
            };
            let was_revealed = self.prev_revealed_indices.contains(&i);
            let is_revealed = self.revealed_indices.contains(&i);
            if was_revealed == is_revealed {
                continue;
            }
            affected.push(fold_range.clone());
        }
        affected.sort_by_key(|r| r.start);
        affected
    }

    fn fill_isomorphic_gap(&self, transforms: &mut SumTree<Transform>, target: FoldOffset) {
        let fold_snapshot = &self.snapshot.fold_snapshot;
        let current = FoldOffset(transforms.summary().input.len);
        if target > current {
            push_isomorphic(
                transforms,
                fold_snapshot.text_summary_for_range(
                    current.to_point(fold_snapshot)..target.to_point(fold_snapshot),
                ),
            );
        }
    }

    fn build_region_filled(
        &self,
        transforms: &mut SumTree<Transform>,
        resolved: &[(Range<FoldOffset>, SharedString)],
        region_start: FoldOffset,
        region_end: FoldOffset,
    ) {
        self.build_region(transforms, resolved, region_start, region_end);
        self.fill_isomorphic_gap(transforms, region_end);
    }

    fn build_region(
        &self,
        transforms: &mut SumTree<Transform>,
        resolved: &[(Range<FoldOffset>, SharedString)],
        start: FoldOffset,
        end: FoldOffset,
    ) {
        let fold_snapshot = &self.snapshot.fold_snapshot;
        let first = resolved.partition_point(|(range, _)| range.end <= start);
        for (range, replacement) in &resolved[first..] {
            if range.start >= end {
                break;
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

impl ConcealSnapshot {
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
        let line_start = ConcealPoint::new(row, 0).to_offset(self).0.0;
        let line_end = if row >= self.max_point().row() {
            self.len().0.0
        } else {
            ConcealPoint::new(row + 1, 0)
                .to_offset(self)
                .0
                .0
                .saturating_sub(1)
        };
        line_end.saturating_sub(line_start) as u32
    }

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
                transforms: &self.transforms,
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
                max_fold_offset: fold_end,
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

        let max_fold_offset = {
            let mut end_cursor = self
                .transforms
                .cursor::<Dimensions<ConcealOffset, FoldOffset>>(());
            end_cursor.seek(&range.end, Bias::Right);
            if let Some(transform) = end_cursor.item() {
                if transform.is_concealment() {
                    end_cursor.end().1
                } else {
                    let overshoot = range.end - end_cursor.start().0;
                    end_cursor.start().1 + overshoot
                }
            } else {
                FoldOffset(self.fold_snapshot.len().0)
            }
        };

        ConcealChunks {
            passthrough: false,
            transforms: &self.transforms,
            transform_cursor,
            fold_chunks: self.fold_snapshot.chunks(
                fold_start..max_fold_offset,
                language_aware,
                highlights,
            ),
            fold_chunk: None,
            fold_offset: fold_start,
            output_offset: range.start,
            max_output_offset: range.end,
            max_fold_offset,
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

impl ConcealChunks<'_> {
    pub(crate) fn seek(&mut self, range: Range<ConcealOffset>) {
        if self.passthrough {
            let fold_end = FoldOffset(range.end.0);
            self.fold_chunks.seek(FoldOffset(range.start.0)..fold_end);
            self.output_offset = range.start;
            self.max_output_offset = range.end;
            self.max_fold_offset = fold_end;
            return;
        }

        self.transform_cursor.seek(&range.start, Bias::Right);

        let fold_start = {
            let overshoot = range.start - self.transform_cursor.start().0;
            self.transform_cursor.start().1 + overshoot
        };

        let mut end_cursor = self
            .transforms
            .cursor::<Dimensions<ConcealOffset, FoldOffset>>(());
        end_cursor.seek(&range.end, Bias::Right);
        self.max_fold_offset = if let Some(transform) = end_cursor.item() {
            if transform.is_concealment() {
                end_cursor.end().1
            } else {
                let overshoot = range.end - end_cursor.start().0;
                end_cursor.start().1 + overshoot
            }
        } else {
            FoldOffset(self.transforms.summary().input.len)
        };

        // Seek fold_chunks to the full range upfront so the iterator can
        // advance through concealments without expensive per-concealment
        // re-seeks that cascade through all underlying map layers.
        self.fold_chunks.seek(fold_start..self.max_fold_offset);
        self.fold_chunk = None;
        self.fold_offset = fold_start;
        self.output_offset = range.start;
        self.max_output_offset = range.end;
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
            let text = replacement.as_ref();

            let conceal_fold_end = self.transform_cursor.end().1;

            // Grab highlight info from the fold chunk at the current
            // position. Only the scalar style fields are needed — no
            // need to clone the whole Chunk.
            if self.fold_chunk.is_none() {
                let offset = self.fold_offset;
                self.fold_chunk = self.fold_chunks.next().map(|c| (offset, c));
            }
            let highlight = self.fold_chunk.as_ref().map(|(_, c)| {
                (
                    c.syntax_highlight_id,
                    c.highlight_style,
                    c.diagnostic_severity,
                    c.is_unnecessary,
                    c.underline,
                )
            });

            // With language_aware=true, fold chunks are split at syntax token
            // boundaries so a concealment may span multiple chunks (e.g.
            // "not in" → ["not", " ", "in"]). Consume all of them.
            let mut fold_consumed = self
                .fold_chunk
                .as_ref()
                .map_or(self.fold_offset, |(start, c)| *start + c.text.len());

            self.fold_chunk = self
                .fold_chunk
                .take()
                .filter(|(start, c)| *start + c.text.len() > conceal_fold_end);

            while fold_consumed < conceal_fold_end {
                let chunk_fold_start = fold_consumed;
                match self.fold_chunks.next() {
                    Some(c) => {
                        fold_consumed += c.text.len();
                        self.fold_chunk = if fold_consumed > conceal_fold_end {
                            Some((chunk_fold_start, c))
                        } else {
                            None
                        };
                    }
                    None => break,
                }
            }

            self.fold_offset = conceal_fold_end;
            self.output_offset.0 += text.len();
            self.transform_cursor.next();

            return Some(
                if let Some((syn_id, hl_style, diag, unnecessary, underline)) = highlight {
                    Chunk {
                        text,
                        syntax_highlight_id: syn_id,
                        highlight_style: hl_style,
                        diagnostic_severity: diag,
                        is_unnecessary: unnecessary,
                        underline,
                        ..Default::default()
                    }
                } else {
                    Chunk {
                        text,
                        ..Default::default()
                    }
                },
            );
        }

        if self.fold_chunk.is_none() {
            let chunk_offset = self.fold_offset;
            self.fold_chunk = self.fold_chunks.next().map(|chunk| (chunk_offset, chunk));
        }

        // Borrow the buffered chunk to extract all Copy fields, avoiding
        // a clone of the full Chunk (which includes an Arc in renderer).
        let (chunk_start, chunk) = self.fold_chunk.as_ref()?;
        let chunk_start = *chunk_start;
        let chunk_text = chunk.text;
        let chunk_end = chunk_start + chunk_text.len();
        let transform_end = self.transform_cursor.end().1;
        let end = chunk_end.min(transform_end);

        let bit_start = self.fold_offset - chunk_start;
        let bit_end = end - chunk_start;
        let text = &chunk_text[bit_start..bit_end];
        let mask = 1u128
            .unbounded_shl((bit_end - bit_start) as u32)
            .wrapping_sub(1);
        let tabs = (chunk.tabs >> bit_start) & mask;
        let chars = (chunk.chars >> bit_start) & mask;
        let newlines = (chunk.newlines >> bit_start) & mask;
        let syntax_highlight_id = chunk.syntax_highlight_id;
        let highlight_style = chunk.highlight_style;
        let diagnostic_severity = chunk.diagnostic_severity;
        let is_unnecessary = chunk.is_unnecessary;
        let is_tab = chunk.is_tab;
        let is_inlay = chunk.is_inlay;
        let underline = chunk.underline;

        if end == transform_end {
            self.transform_cursor.next();
        }
        let renderer = if end == chunk_end {
            self.fold_chunk.take().and_then(|(_, c)| c.renderer)
        } else {
            self.fold_chunk.as_ref().and_then(|(_, c)| c.renderer.clone())
        };

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
            syntax_highlight_id,
            highlight_style,
            diagnostic_severity,
            is_unnecessary,
            is_tab,
            is_inlay,
            underline,
            renderer,
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

fn build_transforms_from_resolved(
    transforms: &mut SumTree<Transform>,
    fold_snapshot: &FoldSnapshot,
    resolved: &[(Range<FoldOffset>, SharedString)],
) {
    let mut offset = FoldOffset(MultiBufferOffset(0));
    for (range, replacement) in resolved {
        if range.start > offset {
            push_isomorphic(
                transforms,
                fold_snapshot.text_summary_for_range(
                    offset.to_point(fold_snapshot)..range.start.to_point(fold_snapshot),
                ),
            );
        }

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
        push_isomorphic(
            transforms,
            fold_snapshot.text_summary_for_range(
                offset.to_point(fold_snapshot)..total.to_point(fold_snapshot),
            ),
        );
    }

    if transforms.is_empty() {
        push_isomorphic(transforms, fold_snapshot.text_summary());
    }
}

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
    use rand::rngs::StdRng;
    use settings::SettingsStore;

    fn init_test(cx: &mut gpui::App) {
        let settings_store = SettingsStore::test(cx);
        cx.set_global(settings_store);
    }

    fn setup(
        text: &str,
        cx: &mut gpui::App,
    ) -> (ConcealMap, FoldSnapshot, multi_buffer::MultiBufferSnapshot) {
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

        conceal_map.set_revealed_indices(HashSet::from_iter([0]));
        let (snapshot, _) = conceal_map.read(fold_snapshot.clone(), vec![]);
        assert_eq!(snapshot.text(), "x = 2\nlambda x: x + 1\n");

        // Re-conceal (cursor moved away)
        conceal_map.set_revealed_indices(HashSet::default());
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

    #[gpui::test]
    fn test_empty_buffer(cx: &mut gpui::App) {
        let (mut conceal_map, _, buffer_snapshot) = setup("", cx);

        // Setting concealments on an empty buffer should be a no-op.
        let (snapshot, edits) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(0)),
            "λ".into(),
        )]);
        assert_eq!(snapshot.text(), "");
        assert!(edits.is_empty());
        assert_eq!(snapshot.max_point(), ConcealPoint::new(0, 0));
        assert_eq!(snapshot.len(), ConcealOffset(O(0)));
    }

    #[gpui::test]
    fn test_concealment_at_buffer_boundaries(cx: &mut gpui::App) {
        // Concealment at the very start.
        let (mut conceal_map, _, buffer_snapshot) = setup("hello world", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(5)),
            "hi".into(),
        )]);
        assert_eq!(snapshot.text(), "hi world");

        // Concealment at the very end.
        let (mut conceal_map, _, buffer_snapshot) = setup("hello world", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(11)),
            "🌍".into(),
        )]);
        assert_eq!(snapshot.text(), "hello 🌍");

        // Concealment spanning the entire buffer.
        let (mut conceal_map, _, buffer_snapshot) = setup("hello", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(0))..buffer_snapshot.anchor_before(O(5)),
            "hi".into(),
        )]);
        assert_eq!(snapshot.text(), "hi");
    }

    #[gpui::test]
    fn test_expanding_concealment(cx: &mut gpui::App) {
        // Replacement longer than the original text.
        let (mut conceal_map, _, buffer_snapshot) = setup("a != b", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(2))..buffer_snapshot.anchor_before(O(4)),
            "not_equal_to".into(),
        )]);
        assert_eq!(snapshot.text(), "a not_equal_to b");

        let summary =
            snapshot.text_summary_for_range(ConcealPoint::new(0, 0)..snapshot.max_point());
        assert_eq!(summary.len, O("a not_equal_to b".len()));
    }

    #[gpui::test]
    fn test_clip_point_with_bias(cx: &mut gpui::App) {
        // "hello != world" -> "hello ≠ world"
        //  offsets: h=0 e=1 l=2 l=3 o=4 ' '=5 ≠=6..9 ' '=9 w=10 ...
        let (mut conceal_map, _, buffer_snapshot) = setup("hello != world", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "hello ≠ world");

        // Point inside the concealment with Left bias -> snaps to start.
        let inside = ConcealPoint::new(0, 7); // middle of "≠" (3-byte char)
        assert_eq!(
            snapshot.clip_point(inside, Bias::Left),
            ConcealPoint::new(0, 6),
        );

        // Point inside the concealment with Right bias -> snaps to end.
        assert_eq!(
            snapshot.clip_point(inside, Bias::Right),
            ConcealPoint::new(0, 9),
        );

        // Point outside concealments clips normally.
        let before = ConcealPoint::new(0, 3);
        assert_eq!(snapshot.clip_point(before, Bias::Left), before);
        assert_eq!(snapshot.clip_point(before, Bias::Right), before);

        // Point past max_point clips to max_point.
        let past_end = ConcealPoint::new(0, 100);
        assert_eq!(
            snapshot.clip_point(past_end, Bias::Left),
            snapshot.max_point()
        );
    }

    #[gpui::test]
    fn test_line_len_with_concealment(cx: &mut gpui::App) {
        // "hello != world\nfoo" -> "hello ≠ world\nfoo"
        let (mut conceal_map, _, buffer_snapshot) = setup("hello != world\nfoo", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "hello ≠ world\nfoo");

        // "hello ≠ world" = 5+1+3+1+5 = 15 bytes (≠ is 3 bytes UTF-8)
        assert_eq!(snapshot.line_len(0), "hello ≠ world".len() as u32);
        // Last row: "foo" = 3 bytes
        assert_eq!(snapshot.line_len(1), 3);
    }

    #[gpui::test]
    fn test_line_len_last_row_no_trailing_newline(cx: &mut gpui::App) {
        // No trailing newline — last row has content.
        let (mut conceal_map, _, buffer_snapshot) = setup("a != b", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(2))..buffer_snapshot.anchor_before(O(4)),
            "≠".into(),
        )]);
        assert_eq!(snapshot.text(), "a ≠ b");
        assert_eq!(snapshot.line_len(0), "a ≠ b".len() as u32);
    }

    #[gpui::test]
    fn test_conceal_rows_iterator(cx: &mut gpui::App) {
        // Three rows, concealment on second row.
        let (mut conceal_map, _, buffer_snapshot) = setup("aaa\nlambda\nccc\n", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![(
            buffer_snapshot.anchor_after(O(4))..buffer_snapshot.anchor_before(O(10)),
            "λ".into(),
        )]);
        assert_eq!(snapshot.text(), "aaa\nλ\nccc\n");

        let mut rows = snapshot.row_infos(0);
        // Row 0 -> buffer_row 0
        let r0 = rows.next().expect("row 0");
        assert_eq!(r0.buffer_row, Some(0));
        // Row 1 -> buffer_row 1
        let r1 = rows.next().expect("row 1");
        assert_eq!(r1.buffer_row, Some(1));
        // Row 2 -> buffer_row 2
        let r2 = rows.next().expect("row 2");
        assert_eq!(r2.buffer_row, Some(2));
        // Row 3 -> buffer_row 3 (trailing newline)
        let r3 = rows.next().expect("row 3");
        assert_eq!(r3.buffer_row, Some(3));
        assert!(rows.next().is_none());
    }

    #[gpui::test]
    fn test_chunks_seek(cx: &mut gpui::App) {
        // "hello != world == end" -> "hello ≠ world ≡ end"
        let (mut conceal_map, _, buffer_snapshot) = setup("hello != world == end", cx);
        let (snapshot, _) = conceal_map.set_concealments(vec![
            (
                buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(8)),
                "≠".into(),
            ),
            (
                buffer_snapshot.anchor_after(O(15))..buffer_snapshot.anchor_before(O(17)),
                "≡".into(),
            ),
        ]);
        let full_text = snapshot.text();
        assert_eq!(full_text, "hello ≠ world ≡ end");

        // Seek to an offset past the first concealment and collect the rest.
        let mid = ConcealOffset(O("hello ≠ wor".len()));
        let chunks = snapshot.chunks(mid..snapshot.len(), false, Highlights::default());
        let after_seek: String = chunks.map(|c| c.text.to_string()).collect();
        assert_eq!(after_seek, "ld ≡ end");

        // Seek again to the second concealment.
        let mid2 = ConcealOffset(O("hello ≠ world ".len()));
        let chunks = snapshot.chunks(mid2..snapshot.len(), false, Highlights::default());
        let after_seek2: String = chunks.map(|c| c.text.to_string()).collect();
        assert_eq!(after_seek2, "≡ end");
    }

    #[gpui::test(iterations = 100)]
    fn test_random_concealments(cx: &mut gpui::App, mut rng: StdRng) {
        use rand::Rng;

        init_test(cx);

        // Generate random text (3-10 lines, 5-20 chars each).
        let num_lines = rng.random_range(1..=10);
        let mut text = String::new();
        for i in 0..num_lines {
            let len = rng.random_range(5..=20);
            for _ in 0..len {
                let c = rng.random_range(b'a'..=b'z') as char;
                text.push(c);
            }
            if i < num_lines - 1 {
                text.push('\n');
            }
        }

        let buffer = MultiBuffer::build_simple(&text, cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (mut inlay_map, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (mut fold_map, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot.clone());

        // Generate random non-overlapping concealments.
        let text_len = text.len();
        let num_concealments = rng.random_range(0..=6);
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for _ in 0..num_concealments {
            if text_len < 3 {
                break;
            }
            let start = rng.random_range(0..text_len.saturating_sub(2));
            let max_len = (text_len - start).min(5);
            if max_len < 1 {
                continue;
            }
            let len = rng.random_range(1..=max_len);
            let end = start + len;
            // Skip if overlapping an existing range.
            if ranges.iter().any(|&(s, e)| start < e && end > s) {
                continue;
            }
            ranges.push((start, end));
        }
        ranges.sort_by_key(|&(s, _)| s);

        let concealments: Vec<_> = ranges
            .iter()
            .map(|&(start, end)| {
                (
                    buffer_snapshot.anchor_after(O(start))..buffer_snapshot.anchor_before(O(end)),
                    SharedString::from("•"),
                )
            })
            .collect();

        let (snapshot, _) = conceal_map.set_concealments(concealments.clone());

        // Verify: concealed text length matches chunks.
        let text_from_chunks = snapshot.text();
        let expected_len = snapshot.len();
        assert_eq!(
            text_from_chunks.len(),
            expected_len.0.0,
            "text len mismatch after set_concealments"
        );

        // Verify point round-trips.
        let max_point = snapshot.max_point();
        for row in 0..=max_point.row() {
            let line_len = snapshot.line_len(row);
            let start = ConcealPoint::new(row, 0);
            let end = ConcealPoint::new(row, line_len);
            let start_offset = start.to_offset(&snapshot);
            let end_offset = end.to_offset(&snapshot);
            assert!(
                end_offset >= start_offset,
                "line {row}: end offset < start offset"
            );
            let back = start_offset.to_point(&snapshot);
            assert_eq!(back, start, "round-trip offset→point for row {row}");
        }

        // Randomly reveal some concealments.
        if !concealments.is_empty() {
            let reveal_count = rng.random_range(0..=concealments.len());
            let revealed: HashSet<usize> = (0..concealments.len())
                .filter(|_| rng.random_bool(0.5))
                .take(reveal_count)
                .collect();
            conceal_map.set_revealed_indices(revealed);
            let (snapshot, _) = conceal_map.read(fold_snapshot, vec![]);

            // After reveal, text should still be self-consistent.
            let text_after_reveal = snapshot.text();
            assert_eq!(
                text_after_reveal.len(),
                snapshot.len().0.0,
                "text len mismatch after reveal"
            );
        }

        // Simulate a buffer edit and sync through the pipeline.
        let subscription = buffer.update(cx, |buffer, _| buffer.subscribe());
        let edit_start = rng.random_range(0..=text_len);
        let edit_end = rng.random_range(edit_start..=text_len);
        let insert_len = rng.random_range(0..=5);
        let insert: String = (0..insert_len)
            .map(|_| rng.random_range(b'a'..=b'z') as char)
            .collect();

        buffer.update(cx, |buffer, cx| {
            buffer.edit([(O(edit_start)..O(edit_end), insert.as_str())], None, cx);
        });
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let edits = subscription.consume().into_inner();

        let (inlay_snapshot, inlay_edits) = inlay_map.sync(buffer_snapshot, edits);
        let (fold_snapshot, fold_edits) = fold_map.read(inlay_snapshot, inlay_edits);
        let (snapshot, _) = conceal_map.read(fold_snapshot, fold_edits);

        // Final consistency: text length matches len().
        let final_text = snapshot.text();
        assert_eq!(
            final_text.len(),
            snapshot.len().0.0,
            "text len mismatch after edit sync"
        );
    }
}
