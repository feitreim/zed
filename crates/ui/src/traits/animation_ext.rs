use std::time::Duration;

use gpui::{Animation, AnimationElement, AnimationExt, Transformation, percentage};

use crate::{prelude::*, traits::transformable::Transformable};

/// An extension trait for adding common animations to animatable components.
pub trait CommonAnimationExt: AnimationExt {
    /// Render this component as rotating over the given duration.
    ///
    /// NOTE: This method uses the location of the caller to generate an ID for this state.
    ///       If this is not sufficient to identify your state (e.g. you're rendering a list item),
    ///       you can provide a custom ElementID using the `use_keyed_rotate_animation` method.
    #[track_caller]
    fn with_rotate_animation(self, duration: u64) -> AnimationElement<Self>
    where
        Self: Transformable + Sized,
    {
        self.with_keyed_rotate_animation(
            ElementId::CodeLocation(*std::panic::Location::caller()),
            duration,
        )
    }

    /// Render this component as rotating with the given element ID over the given duration.
    fn with_keyed_rotate_animation(
        self,
        id: impl Into<ElementId>,
        duration: u64,
    ) -> AnimationElement<Self>
    where
        Self: Transformable + Sized,
    {
        // Advancing the rotation ~30 times per second is visually smooth for
        // an icon-sized spinner and keeps it from forcing a full-window
        // redraw on every vsync for as long as it's shown.
        let steps = (duration.saturating_mul(30)).max(1) as usize;
        self.with_animation(
            id,
            Animation::new(Duration::from_secs(duration))
                .repeat()
                .with_steps(steps),
            |component, delta| component.transform(Transformation::rotate(percentage(delta))),
        )
    }
}

impl<T: AnimationExt> CommonAnimationExt for T {}
