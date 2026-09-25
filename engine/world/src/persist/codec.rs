//! Little-endian byte encoding and CRC-32 (IEEE 802.3, the zlib/PNG polynomial), both in-house so
//! the crate keeps no external dependencies.

use super::PersistError;

const CRC_TABLE: [u32; 256] = crc_table();

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

/// Continues a CRC: `crc32_update(crc32(a), b) == crc32(a ++ b)`.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

#[derive(Default)]
pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// Exact: the bit pattern is stored.
    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
}

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Names the structure being read, for error messages.
    what: &'static str,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8], what: &'static str) -> Self {
        Self { buf, pos: 0, what }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8], PersistError> {
        if self.remaining() < n {
            return Err(PersistError::Truncated(self.what));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], PersistError> {
        Ok(self.take(N)?.try_into().expect("length checked"))
    }

    pub fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, PersistError> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    pub fn u32(&mut self) -> Result<u32, PersistError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    pub fn u64(&mut self) -> Result<u64, PersistError> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    pub fn i32(&mut self) -> Result<i32, PersistError> {
        Ok(i32::from_le_bytes(self.array()?))
    }
    pub fn f32(&mut self) -> Result<f32, PersistError> {
        Ok(f32::from_bits(self.u32()?))
    }
    pub fn tag(&mut self) -> Result<[u8; 4], PersistError> {
        self.array()
    }

    /// Fails unless everything was consumed.
    pub fn finish(self) -> Result<(), PersistError> {
        if self.remaining() != 0 {
            return Err(PersistError::Malformed(format!("{}: {} trailing bytes", self.what, self.remaining())));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32_update(crc32(b"1234"), b"56789"), crc32(b"123456789"));
    }

    #[test]
    fn reader_round_trips_and_reports_truncation() {
        let mut w = Writer::default();
        w.u8(7);
        w.u16(0xBEEF);
        w.i32(-5);
        w.u64(u64::MAX - 1);
        w.f32(-0.0);
        let mut r = Reader::new(&w.buf, "test");
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u16().unwrap(), 0xBEEF);
        assert_eq!(r.i32().unwrap(), -5);
        assert_eq!(r.u64().unwrap(), u64::MAX - 1);
        assert_eq!(r.f32().unwrap().to_bits(), (-0.0f32).to_bits());
        assert!(matches!(r.u8(), Err(PersistError::Truncated("test"))));
    }
}
