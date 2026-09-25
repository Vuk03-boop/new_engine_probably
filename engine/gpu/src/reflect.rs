//! Host/shader layout agreement from `slangc -reflection-json` (ADR-0001: reflection comes from
//! `slangc` output). A shader module is only turned into a pipeline after its reflection matches
//! the layout the host code declares: binding names and indices, and every struct field's offset
//! and size.
//!
//! The JSON reader is minimal (objects, arrays, strings, numbers, booleans, null), enough for
//! reflection files; it rejects anything else rather than guessing.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { s: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> &[Json] {
        match self {
            Json::Arr(a) => a,
            _ => &[],
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut m = BTreeMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    let v = self.value()?;
                    if m.insert(k.clone(), v).is_some() {
                        return Err(format!("duplicate key {k}"));
                    }
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(m));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Ok(Json::Null)
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.i]).unwrap();
                t.parse().map(Json::Num).map_err(|_| format!("bad number {t}"))
            }
            _ => Err(format!("unexpected input at byte {}", self.i)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') {
            return Err(format!("expected string at byte {}", self.i));
        }
        self.i += 1;
        let mut out = Vec::new();
        while let Some(&c) = self.s.get(self.i) {
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        _ => return Err(format!("unsupported escape \\{}", e as char)),
                    }
                }
                _ => out.push(c),
            }
        }
        Err("unterminated string".into())
    }
}

/// One struct field the host expects: name, byte offset, byte size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: &'static str,
    pub offset: u64,
    pub size: u64,
}

/// What the host declares about one shader parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Param {
    /// A descriptor at `binding` (set 0). `element` is the struct layout of a structured buffer's
    /// element, or empty for a scalar/vector element.
    Descriptor { name: &'static str, binding: u64, element: Vec<Field> },
    /// The push-constant block and its fields.
    PushConstants { name: &'static str, fields: Vec<Field> },
}

fn fields_of(ty: &Json) -> Vec<(String, u64, u64)> {
    ty.get("fields")
        .map(|f| {
            f.as_arr()
                .iter()
                .map(|x| {
                    let b = x.get("binding");
                    (
                        x.get("name").and_then(Json::as_str).unwrap_or("").to_string(),
                        b.and_then(|b| b.get("offset")).and_then(Json::as_u64).unwrap_or(u64::MAX),
                        b.and_then(|b| b.get("size")).and_then(Json::as_u64).unwrap_or(u64::MAX),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Checks a module's reflection against the host's declaration. Every declared parameter must be
/// present with the same binding and field layout, and the module must have no undeclared parameter.
pub fn check(reflection: &str, expected: &[Param]) -> Result<(), Vec<String>> {
    let root = Json::parse(reflection).map_err(|e| vec![format!("reflection JSON: {e}")])?;
    let params = root.get("parameters").map(Json::as_arr).unwrap_or(&[]);
    let mut errors = Vec::new();
    let by_name: BTreeMap<&str, &Json> = params.iter().filter_map(|p| Some((p.get("name")?.as_str()?, p))).collect();
    for e in expected {
        let (name, want_fields) = match e {
            Param::Descriptor { name, element, .. } => (*name, element),
            Param::PushConstants { name, fields } => (*name, fields),
        };
        let Some(p) = by_name.get(name) else {
            errors.push(format!("{name}: not in the shader"));
            continue;
        };
        let binding = p.get("binding");
        let kind = binding.and_then(|b| b.get("kind")).and_then(Json::as_str);
        let ty = p.get("type");
        let got_fields = match e {
            Param::Descriptor { binding: want, .. } => {
                let got = binding.and_then(|b| b.get("index")).and_then(Json::as_u64);
                if kind != Some("descriptorTableSlot") || got != Some(*want) {
                    errors.push(format!("{name}: binding {kind:?} {got:?}, host expects descriptor {want}"));
                }
                ty.and_then(|t| t.get("resultType")).map(fields_of).unwrap_or_default()
            }
            Param::PushConstants { .. } => {
                if kind != Some("pushConstantBuffer") {
                    errors.push(format!("{name}: binding kind {kind:?}, host expects push constants"));
                }
                ty.and_then(|t| t.get("elementType")).map(fields_of).unwrap_or_default()
            }
        };
        let want: Vec<(String, u64, u64)> = want_fields.iter().map(|f| (f.name.to_string(), f.offset, f.size)).collect();
        if got_fields != want {
            errors.push(format!("{name}: shader fields {got_fields:?}, host expects {want:?}"));
        }
    }
    let declared: Vec<&str> = expected
        .iter()
        .map(|e| match e {
            Param::Descriptor { name, .. } | Param::PushConstants { name, .. } => *name,
        })
        .collect();
    for n in by_name.keys() {
        if !declared.contains(n) {
            errors.push(format!("{n}: in the shader but not declared by the host"));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// One entry-point varying: a vertex input or a fragment output, at a location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Varying {
    pub name: String,
    pub location: u64,
    /// Reflection scalar type, e.g. `float32` or `uint32`.
    pub scalar: String,
    pub components: u64,
}

fn varying_of(name: &str, binding: Option<&Json>, ty: Option<&Json>) -> Option<Varying> {
    let location = binding?.get("index")?.as_u64()?;
    let ty = ty?;
    let (scalar, components) = match ty.get("kind")?.as_str()? {
        "scalar" => (ty.get("scalarType")?.as_str()?, 1),
        "vector" => (ty.get("elementType")?.get("scalarType")?.as_str()?, ty.get("elementCount")?.as_u64()?),
        _ => return None,
    };
    Some(Varying { name: name.to_string(), location, scalar: scalar.to_string(), components })
}

/// The varying inputs (`varyingInput`) and outputs (`varyingOutput`: a bare result, or the result
/// struct's fields) of the module's single entry point. System values (no location) are left out.
pub fn varyings(reflection: &str) -> Result<(Vec<Varying>, Vec<Varying>), String> {
    let root = Json::parse(reflection)?;
    let eps = root.get("entryPoints").map(Json::as_arr).unwrap_or(&[]);
    let [ep] = eps else { return Err(format!("expected one entry point, found {}", eps.len())) };
    let kind = |b: Option<&Json>| b.and_then(|b| b.get("kind")).and_then(Json::as_str).map(str::to_string);
    let name = |x: &Json| x.get("name").and_then(Json::as_str).unwrap_or("").to_string();
    let mut inputs = Vec::new();
    for p in ep.get("parameters").map(Json::as_arr).unwrap_or(&[]) {
        if kind(p.get("binding")).as_deref() == Some("varyingInput") {
            inputs.extend(varying_of(&name(p), p.get("binding"), p.get("type")));
        }
    }
    let mut outputs = Vec::new();
    // A bare (non-struct) result is one output with its own binding; it has no name ("").
    if let Some(r) = ep.get("result").filter(|r| kind(r.get("binding")).as_deref() == Some("varyingOutput")) {
        outputs.extend(varying_of("", r.get("binding"), r.get("type")));
    }
    if let Some(ty) = ep.get("result").and_then(|r| r.get("type")) {
        for f in ty.get("fields").map(Json::as_arr).unwrap_or(&[]) {
            if kind(f.get("binding")).as_deref() == Some("varyingOutput") {
                outputs.extend(varying_of(&name(f), f.get("binding"), f.get("type")));
            }
        }
    }
    Ok((inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values_and_rejects_garbage() {
        let v = Json::parse(r#"{"a": [1, 2.5, -3e2, true, null, "x\"y"], "b": {}}"#).unwrap();
        let a = v.get("a").unwrap().as_arr();
        assert_eq!(a[0].as_u64(), Some(1));
        assert_eq!(a[2], Json::Num(-300.0));
        assert_eq!(a[5].as_str(), Some("x\"y"));
        assert!(Json::parse(r#"{"a": 1,}"#).is_err());
        assert!(Json::parse(r#"{"a": 1} x"#).is_err());
        assert!(Json::parse(r#"{"a": 1, "a": 2}"#).is_err());
    }
}
