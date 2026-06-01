#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphDirection {
    #[default]
    Outbound,
    Inbound,
    Both,
}
