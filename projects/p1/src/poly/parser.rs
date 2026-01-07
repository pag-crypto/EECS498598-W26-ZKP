// simple little ``recursive''' descent parser for the FromStr impls for
// Multilinear and Univariate (not really recursive bc the grammar isn't recursive)
//
// STUDENTS MAY IGNORE THE CONTENTS OF THIS FILE
//
// MAYBEFIX: this really should be generic over F: Field instead of making the coeff u64
#[derive(Debug)]
pub struct Term {
    pub sign: bool,
    pub coeff: u64,
    pub vars: Vec<(usize, u64)>,
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Parser {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn skip_whitespace(&mut self) {
        self.seek_until(|c| !c.is_ascii_whitespace());
    }

    fn seek_until<F>(&mut self, pred: F)
    where
        F: Fn(u8) -> bool,
    {
        while let Some(c) = self.peek()
            && !pred(c)
        {
            self.advance();
        }
    }

    fn error(&self, msg: &str) -> String {
        format!("{} at position {}", msg, self.pos)
    }

    fn parse_number(&mut self) -> Result<u64, String> {
        self.skip_whitespace();
        let start = self.pos;
        self.seek_until(|c| !c.is_ascii_digit());
        if self.pos > start {
            let s = std::str::from_utf8(&self.input[start..self.pos])
                .map_err(|_| self.error("invalid UTF-8"))?;
            s.parse().map_err(|_| self.error("number overflow"))
        } else {
            Err(self.error("expected number"))
        }
    }

    fn parse_variable(&mut self) -> Result<(usize, u64), String> {
        self.skip_whitespace();
        if self.peek() != Some(b'x') {
            return Err(self.error("expected 'x'"));
        }
        self.advance();

        // Skip optional underscore before subscript
        if self.peek() == Some(b'_') {
            self.advance();
        }

        // Parse optional subscript (variable index)
        let index = if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.parse_number()? as usize
        } else {
            0
        };

        // Parse optional exponent
        self.skip_whitespace();
        let exponent = if self.peek() == Some(b'^') {
            self.advance();
            self.parse_number()?
        } else {
            1
        };

        Ok((index, exponent))
    }

    fn parse_factor(&mut self) -> Result<(Option<u64>, Option<(usize, u64)>), String> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'x') => {
                let var = self.parse_variable()?;
                Ok((None, Some(var)))
            }
            Some(c) if c.is_ascii_digit() => {
                let num = self.parse_number()?;
                Ok((Some(num), None))
            }
            Some(c) => Err(self.error(&format!("unexpected character '{}'", c as char))),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_term(&mut self) -> Result<Term, String> {
        self.skip_whitespace();

        let mut coeff: u64 = 1;
        let mut vars: Vec<(usize, u64)> = Vec::new();

        // Parse first factor
        let (num, var) = self.parse_factor()?;
        if let Some(n) = num {
            coeff = n;
        }
        if let Some(v) = var {
            vars.push(v);
        }

        // Parse remaining factors separated by '*'
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'*') {
                break;
            }
            self.advance();

            let (num, var) = self.parse_factor()?;
            if let Some(n) = num {
                coeff = coeff
                    .checked_mul(n)
                    .ok_or_else(|| self.error("coefficient overflow"))?;
            }
            if let Some(v) = var {
                vars.push(v);
            }
        }

        vars.sort_unstable_by_key(|(idx, _)| *idx);

        Ok(Term {
            sign: true,
            coeff,
            vars,
        })
    }

    fn parse_poly(&mut self) -> Result<Vec<Term>, String> {
        self.skip_whitespace();

        let mut terms = Vec::new();

        // Handle optional leading sign
        let first_sign = if self.peek() == Some(b'-') {
            self.advance();
            false
        } else if self.peek() == Some(b'+') {
            self.advance();
            true
        } else {
            true
        };

        // Parse first term
        let mut first_term = self.parse_term()?;
        first_term.sign = first_sign;
        terms.push(first_term);

        // Parse remaining terms
        loop {
            self.skip_whitespace();
            let sign = match self.peek() {
                Some(b'+') => {
                    self.advance();
                    true
                }
                Some(b'-') => {
                    self.advance();
                    false
                }
                _ => break,
            };

            let mut term = self.parse_term()?;
            term.sign = sign;
            terms.push(term);
        }

        Ok(terms)
    }
}

pub fn parse_terms(input: &str) -> Result<Vec<Term>, String> {
    let mut parser = Parser::new(input);
    let result = parser.parse_poly()?;
    parser.skip_whitespace();
    if parser.pos == parser.input.len() {
        Ok(result)
    } else {
        Err(parser.error("unexpected trailing input"))
    }
}
