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

pub struct ConcealMap {
    snapshot: ConcealSnapshot,
    concealments: Vec<(Range<Anchor>, SharedString)>,
    revealed_ranges: Vec<Range<Anchor>>,
    revealed_dirty: bool,
}

impl ConcealMap {
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

    pub fn read(
        &mut self,
        fold_snapshot: FoldSnapshot,
        fold_edits: Vec<FoldEdit>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        let edits = self.sync(fold_snapshot, fold_edits);
        (self.snapshot.clone(), edits)
    }

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

    pub fn set_revealed_ranges_deferred(&mut self, revealed_ranges: Vec<Range<Anchor>>) {
        self.revealed_ranges = revealed_ranges;
        self.revealed_dirty = true;
    }

    pub fn set_revealed_ranges(
        &mut self,
        revealed_ranges: Vec<Range<Anchor>>,
    ) -> (ConcealSnapshot, Vec<ConcealEdit>) {
        if self.concealments.is_empty() {
            self.revealed_ranges = revealed_ranges;
            return (self.snapshot.clone(), vec![]);
        }

        let old_snapshot = self.snapshot.clone();
        self.revealed_ranges = revealed_ranges;
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

    fn sync(&mut self, fold_snapshot: FoldSnapshot, fold_edits: Vec<FoldEdit>) -> Vec<ConcealEdit> {
        let reveal_changed = self.revealed_dirty;
        self.revealed_dirty = false;

        if fold_edits.is_empty()
            && self.snapshot.fold_snapshot.version == fold_snapshot.version
            && !reveal_changed
        {
            return Vec::new();
        }

        let old_snapshot = self.snapshot.clone();
        self.snapshot.fold_snapshot = fold_snapshot;

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
            if old_output == new_output {
                // Nothing visually changed — don't bump version.
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
            self.snapshot.version += 1;
            fold_edits
                .into_iter()
                .map(|edit| ConcealEdit {
                    old: ConcealOffset(edit.old.start.0)..ConcealOffset(edit.old.end.0),
                    new: ConcealOffset(edit.new.start.0)..ConcealOffset(edit.new.end.0),
                })
                .collect()
        } else {
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

fn build_transforms(
    transforms: &mut SumTree<Transform>,
    fold_snapshot: &FoldSnapshot,
    concealments: &[(Range<Anchor>, SharedString)],
    revealed_ranges: &[Range<Anchor>],
) {
    let buffer = &fold_snapshot.inlay_snapshot.buffer;

    let mut resolved: Vec<(Range<FoldOffset>, SharedString)> = concealments
        .iter()
        .filter_map(|(range, replacement)| {
            // Skip concealments that overlap with any revealed range.
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

            let start_fold_point = fold_snapshot.to_fold_point(start_inlay_point, Bias::Right);
            let end_fold_point = fold_snapshot.to_fold_point(end_inlay_point, Bias::Left);

            let start_fold = start_fold_point.to_offset(fold_snapshot);
            let end_fold = end_fold_point.to_offset(fold_snapshot);

            if start_fold >= end_fold {
                return None;
            }

            Some((start_fold..end_fold, replacement.clone()))
        })
        .collect();

    resolved.sort_by_key(|(range, _)| range.start);
    resolved.dedup_by(|b, a| b.0.start < a.0.end);

    let mut offset = FoldOffset(MultiBufferOffset(0));
    for (range, replacement) in &resolved {
        if range.start > offset {
            let text_summary = fold_snapshot.text_summary_for_range(
                offset.to_point(fold_snapshot)..range.start.to_point(fold_snapshot),
            );
            push_isomorphic(transforms, text_summary);
        }

        let input_summary = fold_snapshot.text_summary_for_range(
            range.start.to_point(fold_snapshot)..range.end.to_point(fold_snapshot),
        );

        transforms.push(
            Transform {
                summary: TransformSummary {
                    input: input_summary,
                    output: MBTextSummary::from(replacement.as_ref()),
                },
                replacement: Some(replacement.clone()),
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

fn push_isomorphic(transforms: &mut SumTree<Transform>, summary: MBTextSummary) {
    let mut did_merge = false;
    transforms.update_last(
        |last| {
            if !last.is_concealment() {
                last.summary.input += summary;
                last.summary.output += summary;
                did_merge = true;
            }
        },
        (),
    );
    if !did_merge {
        transforms.push(
            Transform {
                summary: TransformSummary {
                    input: summary,
                    output: summary,
                },
                replacement: None,
            },
            (),
        );
    }
}

// --- Coordinate types ---

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

    pub fn to_fold_point(self, snapshot: &ConcealSnapshot) -> FoldPoint {
        let (start, _, _) = snapshot
            .transforms
            .find::<Dimensions<ConcealPoint, FoldPoint>, _>((), &self, Bias::Right);
        let overshoot = self.0 - start.0.0;
        FoldPoint(start.1.0 + overshoot)
    }

    pub fn to_offset(self, snapshot: &ConcealSnapshot) -> ConcealOffset {
        let (start, _, item) = snapshot
            .transforms
            .find::<Dimensions<ConcealPoint, TransformSummary>, _>((), &self, Bias::Right);
        let overshoot = self.0 - start.1.output.lines;
        let mut offset = start.1.output.len;
        if !overshoot.is_zero() {
            if let Some(transform) = item {
                assert!(transform.replacement.is_none());
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

impl<'a> sum_tree::Dimension<'a, TransformSummary> for ConcealPoint {
    fn zero(_cx: ()) -> Self {
        Default::default()
    }

    fn add_summary(&mut self, summary: &'a TransformSummary, _: ()) {
        self.0 += &summary.output.lines;
    }
}

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

pub type ConcealEdit = text::Edit<ConcealOffset>;

// --- Transform types ---

#[derive(Clone, Debug, Default)]
struct Transform {
    summary: TransformSummary,
    replacement: Option<SharedString>,
}

impl Transform {
    fn is_concealment(&self) -> bool {
        self.replacement.is_some()
    }
}

impl sum_tree::Item for Transform {
    type Summary = TransformSummary;

    fn summary(&self, _cx: ()) -> Self::Summary {
        self.summary.clone()
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

// --- Snapshot ---

#[derive(Clone)]
pub struct ConcealSnapshot {
    pub fold_snapshot: FoldSnapshot,
    transforms: SumTree<Transform>,
    pub version: usize,
}

impl Deref for ConcealSnapshot {
    type Target = FoldSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.fold_snapshot
    }
}

impl ConcealSnapshot {
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

pub struct ConcealPointCursor<'transforms> {
    cursor: Cursor<'transforms, 'static, Transform, Dimensions<FoldPoint, ConcealPoint>>,
}

impl ConcealPointCursor<'_> {
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

// --- ConcealOffset methods ---

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

// --- Chunk iterators ---

pub struct ConcealChunks<'a> {
    transform_cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealOffset, FoldOffset>>,
    fold_chunks: FoldChunks<'a>,
    fold_chunk: Option<(FoldOffset, Chunk<'a>)>,
    fold_offset: FoldOffset,
    output_offset: ConcealOffset,
    max_output_offset: ConcealOffset,
    replacement_offset: usize,
}

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

        if let Some(replacement) = &transform.replacement {
            let text = &replacement[self.replacement_offset..];

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

        // Isomorphic: seek fold_chunks to cover this region if needed.
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

        let (chunk_start, chunk) = self.fold_chunk.clone()?;
        let chunk_end = chunk_start + chunk.text.len();
        let transform_end = self.transform_cursor.end().1;
        let end = chunk_end.min(transform_end);

        let bit_start = self.fold_offset - chunk_start;
        let bit_end = end - chunk_start;
        let text = &chunk.text[bit_start..bit_end];
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

// --- Row iterator ---

#[derive(Clone)]
pub struct ConcealRows<'a> {
    cursor: Cursor<'a, 'static, Transform, Dimensions<ConcealPoint, FoldPoint>>,
    input_rows: FoldRows<'a>,
    conceal_point: ConcealPoint,
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
        // "lambda" at start of line 2 — conceal then reveal (simulating cursor entering)
        let text = "x = 2\nlambda x: x + 1\n";
        let buffer = MultiBuffer::build_simple(text, cx);
        let buffer_snapshot = buffer.read(cx).snapshot(cx);
        let (_, inlay_snapshot) = InlayMap::new(buffer_snapshot.clone());
        let (_, fold_snapshot) = super::super::fold_map::FoldMap::new(inlay_snapshot);
        let (mut conceal_map, _) = ConcealMap::new(fold_snapshot);

        // Conceal "lambda" (offset 6..12) → "λ"
        let concealments = vec![(
            buffer_snapshot.anchor_after(O(6))..buffer_snapshot.anchor_before(O(12)),
            SharedString::from("λ"),
        )];
        let (snapshot, _) = conceal_map.set_concealments(concealments);
        assert_eq!(snapshot.text(), "x = 2\nλ x: x + 1\n");

        // Now reveal the line (simulating cursor on the lambda line)
        let revealed =
            vec![buffer_snapshot.anchor_before(O(6))..buffer_snapshot.anchor_after(O(12))];
        let (snapshot, _) = conceal_map.set_revealed_ranges(revealed);
        assert_eq!(snapshot.text(), "x = 2\nlambda x: x + 1\n");

        // Re-conceal (cursor moved away)
        let (snapshot, _) = conceal_map.set_revealed_ranges(vec![]);
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
