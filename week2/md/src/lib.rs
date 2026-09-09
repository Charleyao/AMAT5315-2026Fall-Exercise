
pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_correct() {
        assert_eq!(greeting(), "Hello, world!");
    }
}

