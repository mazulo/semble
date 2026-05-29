use pyo3::prelude::*;

/// Split camelCase/PascalCase/ALLCAPS/digit runs into lowercase sub-tokens.
///
/// Matches the behaviour of the Python `_CAMEL_RE` pattern:
///   `[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+`
///
/// Works entirely on ASCII bytes (the upstream token regex only produces ASCII).
fn camel_split(token: &[u8]) -> Vec<String> {
    let n = token.len();
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;

    while i < n {
        let c = token[i];

        if c.is_ascii_digit() {
            let start = i;
            while i < n && token[i].is_ascii_digit() {
                i += 1;
            }
            // SAFETY: slice contains only ASCII digits, valid UTF-8.
            parts.push(unsafe { std::str::from_utf8_unchecked(&token[start..i]) }.to_string());
        } else if c.is_ascii_uppercase() {
            let start = i;

            // Count consecutive uppercase bytes.
            let mut j = i;
            while j < n && token[j].is_ascii_uppercase() {
                j += 1;
            }
            let upper_count = j - i;

            if upper_count >= 2 && j < n && token[j].is_ascii_lowercase() {
                // Pattern [A-Z]+(?=[A-Z][a-z]): take all-but-last uppercase so the
                // final uppercase letter becomes the start of the next [A-Z]?[a-z]+ match.
                // e.g. "HTTPResponse" -> "HTTP" + "Response"
                i = j - 1;
                // SAFETY: slice is ASCII uppercase only.
                let part = unsafe { std::str::from_utf8_unchecked(&token[start..i]) };
                parts.push(part.to_ascii_lowercase());
            } else if upper_count == 1 {
                // Pattern [A-Z]?[a-z]+: one uppercase followed by lowercase run.
                i += 1;
                while i < n && token[i].is_ascii_lowercase() {
                    i += 1;
                }
                // SAFETY: slice is ASCII only.
                let part = unsafe { std::str::from_utf8_unchecked(&token[start..i]) };
                parts.push(part.to_ascii_lowercase());
            } else {
                // Pattern [A-Z]+: pure uppercase run with no following lowercase.
                i = j;
                // SAFETY: slice is ASCII uppercase only.
                let part = unsafe { std::str::from_utf8_unchecked(&token[start..i]) };
                parts.push(part.to_ascii_lowercase());
            }
        } else if c.is_ascii_lowercase() {
            // Pattern [A-Z]?[a-z]+ without the uppercase prefix.
            let start = i;
            while i < n && token[i].is_ascii_lowercase() {
                i += 1;
            }
            // SAFETY: slice is ASCII lowercase only.
            parts.push(unsafe { std::str::from_utf8_unchecked(&token[start..i]) }.to_string());
        } else {
            i += 1;
        }
    }

    parts
}

/// Mirrors Python's `split_identifier`: returns `[lowered, *parts]` when the
/// token has 2+ parts (snake_case or camelCase), else just `[lowered]`.
fn split_identifier(token: &[u8]) -> Vec<String> {
    let lower: String = token
        .iter()
        .map(|&b| b.to_ascii_lowercase() as char)
        .collect();

    let parts: Vec<String> = if token.contains(&b'_') {
        lower
            .split('_')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    } else {
        camel_split(token)
    };

    if parts.len() >= 2 {
        let mut result = Vec::with_capacity(parts.len() + 1);
        result.push(lower);
        result.extend(parts);
        result
    } else {
        vec![lower]
    }
}

/// Tokenize source text for BM25 indexing.
///
/// Mirrors `tokens.py:tokenize`: extracts `[a-zA-Z_][a-zA-Z0-9_]*` tokens,
/// then expands each via `split_identifier` (camelCase / snake_case splitting).
///
/// Non-ASCII bytes are treated as delimiters and never included in tokens, which
/// matches the Python `_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")` behaviour.
#[pyfunction]
pub fn tokenize(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut result: Vec<String> = Vec::new();
    let mut i = 0;

    while i < n {
        let c = bytes[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            result.extend(split_identifier(&bytes[start..i]));
        } else {
            i += 1;
        }
    }

    result
}

#[pymodule]
fn semble_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str) -> Vec<String> {
        split_identifier(s.as_bytes())
    }

    #[test]
    fn test_simple() {
        assert_eq!(tok("simple"), vec!["simple"]);
    }

    #[test]
    fn test_snake_case() {
        assert_eq!(tok("my_func"), vec!["my_func", "my", "func"]);
        assert_eq!(tok("get_http_response"), vec!["get_http_response", "get", "http", "response"]);
    }

    #[test]
    fn test_camel_case() {
        assert_eq!(tok("HandlerStack"), vec!["handlerstack", "handler", "stack"]);
        assert_eq!(tok("getHTTPResponse"), vec!["gethttpresponse", "get", "http", "response"]);
        assert_eq!(tok("XMLParser"), vec!["xmlparser", "xml", "parser"]);
    }

    #[test]
    fn test_all_upper() {
        assert_eq!(tok("HTTP"), vec!["http"]);
        assert_eq!(tok("XML"), vec!["xml"]);
    }

    #[test]
    fn test_digits() {
        assert_eq!(tok("myVar2Name"), vec!["myvar2name", "my", "var", "2", "name"]);
    }

    #[test]
    fn test_tokenize_text() {
        let result = tokenize("def getHTTPResponse(url: str):");
        assert!(result.contains(&"gethttpresponse".to_string()));
        assert!(result.contains(&"get".to_string()));
        assert!(result.contains(&"http".to_string()));
        assert!(result.contains(&"response".to_string()));
        assert!(result.contains(&"url".to_string()));
        assert!(result.contains(&"str".to_string()));
    }
}
