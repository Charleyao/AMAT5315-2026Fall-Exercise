//! Selectable force methods for the LJ fluid: the naive O(N^2) pair loop and a
//! cell list (Part 5). `ForceMethod` is the run-level switch; `CellList`
//! (Task 2) implements the neighbor search.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ForceMethod {
    #[default]
    Naive,
    Cells,
}

impl ForceMethod {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "naive" => Ok(Self::Naive),
            "cells" => Ok(Self::Cells),
            other => Err(format!("expected naive or cells, got {other:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }
}
