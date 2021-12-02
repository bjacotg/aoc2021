use std::panic;

use clap::{Parser};

#[derive(Parser)]
#[clap(version = "1.0", author = "Kevin K. <kbknapp@gmail.com>")]
struct Opts {
    #[clap(short, long)]
    problem: i32,
}

fn main() {
   let opts = Opts::parse();
   match opts.problem {
       1 => unimplemented!("Not there yet"),
       _ => panic!("Unexpected number provided")
   }
}
