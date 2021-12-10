use clap::Parser;
use core::num;
use regex::Regex;
use std::{
    cmp::{max, min},
    collections::{BTreeSet, HashMap, HashSet},
    fs,
    hash::Hash,
    panic,
    slice::SliceIndex,
};

#[derive(Parser)]
#[clap(version = "1.0", author = "Kevin K. <kbknapp@gmail.com>")]
struct Opts {
    #[clap(short, long)]
    problem: i32,
}

fn main() {
    let opts = Opts::parse();
    let file = format!("input/{}.txt", opts.problem);
    let contents = fs::read_to_string(file).expect("Unable to read file");
    let function = match opts.problem {
        1 => func1,
        2 => func2,
        3 => func3,
        4 => func4,
        5 => func5,
        6 => func6,
        7 => func7,
        8 => func8,
        9 => func9,
        10 => func10,
        _ => panic!("Unexpected number provided"),
    };

    let (first, second) = function(&contents);

    println!("Result first {:?} second {:?}", first, second);
}

type Output = (Option<usize>, Option<usize>);

fn func1(input: &str) -> Output {
    let measurement = input
        .lines()
        .map(|s| s.parse::<i32>().unwrap())
        .collect::<Vec<_>>();
    let increasing_count = |seq: &[_]| {
        seq.iter()
            .zip(seq.iter().skip(1))
            .filter(|(f, s)| f < s)
            .count()
    };
    let first = increasing_count(&measurement);
    let sliding_sum = {
        let i1 = measurement.iter();
        let i2 = measurement.iter().skip(1);
        let i3 = measurement.iter().skip(2);
        i1.zip(i2)
            .zip(i3)
            .map(|((i, j), k)| i + j + k)
            .collect::<Vec<_>>()
    };
    let second = increasing_count(&sliding_sum);

    let second_smarter = measurement
        .iter()
        .zip(measurement.iter().skip(3))
        .filter(|(f, s)| f < s)
        .count();
    assert_eq!(second, second_smarter);

    (Some(first), Some(second))
}

fn func2(input: &str) -> Output {
    enum Direction {
        Forward(i32),
        Down(i32),
        Up(i32),
    }
    let re = Regex::new(r"^([a-z]+) (\d+)$").expect("Bad regex");
    let instructions = input
        .lines()
        .map(|instruction| {
            let cap = re.captures(instruction).expect("No match");
            let count = cap[2].parse().unwrap();
            match &cap[1] {
                "down" => Direction::Down(count),
                "forward" => Direction::Forward(count),
                "up" => Direction::Up(count),
                _ => panic!("Unexpected instruction: {}", &cap[1]),
            }
        })
        .collect::<Vec<_>>();

    let end_position = instructions
        .iter()
        .fold((0, 0), |(hpos, depth), instruction| match instruction {
            Direction::Forward(i) => (hpos + i, depth),
            Direction::Down(i) => (hpos, depth + i),
            Direction::Up(i) => (hpos, depth - i),
        });
    let actual_position =
        instructions.iter().fold(
            (0, 0, 0),
            |(hpos, depth, aim), instruction| match instruction {
                Direction::Forward(i) => (hpos + i, depth + i * aim, aim),
                Direction::Down(i) => (hpos, depth, aim + i),
                Direction::Up(i) => (hpos, depth, aim - i),
            },
        );

    (
        Some((end_position.0 * end_position.1) as usize),
        Some((actual_position.0 * actual_position.1) as usize),
    )
}

fn func3(input: &str) -> Output {
    let lines = input.lines().collect::<Vec<_>>();
    let count = {
        let mut count = vec![0; lines[0].len()];
        for element in &lines {
            element
                .chars()
                .zip(count.iter_mut())
                .for_each(|(e, c)| *c += if e == '1' { 1 } else { 0 });
        }
        count
    };
    let half_point = lines.len() as i32 / 2;
    let gamma = count
        .iter()
        .map(|i| if i > &half_point { 1 } else { 0 })
        .fold(0, |acc, bit| 2 * acc + bit);
    let epsilon = count
        .iter()
        .map(|i| if i > &half_point { 0 } else { 1 })
        .fold(0, |acc, bit| 2 * acc + bit);

    let oxy = {
        let mut potential = lines.clone();
        for index in 0..lines[0].len() {
            if potential.len() == 1 {
                break;
            }
            let total_count = potential.len();
            let one_count = potential
                .iter()
                .filter(|s| s.as_bytes()[index] as char == '1')
                .count();
            potential = potential
                .into_iter()
                .filter(|p| {
                    p.as_bytes()[index] as char
                        == if 2 * one_count >= total_count {
                            '1'
                        } else {
                            '0'
                        }
                })
                .collect();
        }
        potential[0]
            .chars()
            .fold(0, |acc, bit| 2 * acc + if bit == '1' { 1 } else { 0 })
    };

    let co2 = {
        let mut potential = lines.clone();
        for index in 0..lines[0].len() {
            if potential.len() == 1 {
                break;
            }
            let total_count = potential.len();
            let one_count = potential
                .iter()
                .filter(|s| s.as_bytes()[index] as char == '1')
                .count();
            potential = potential
                .into_iter()
                .filter(|p| {
                    p.as_bytes()[index] as char
                        == if 2 * one_count < total_count {
                            '1'
                        } else {
                            '0'
                        }
                })
                .collect();
        }
        potential[0]
            .chars()
            .fold(0, |acc, bit| 2 * acc + if bit == '1' { 1 } else { 0 })
    };
    (Some(gamma * epsilon), Some(oxy * co2))
}

fn func4(input: &str) -> Output {
    let (numbers, grids) = {
        let lines = input.lines().collect::<Vec<_>>();
        let grids: Vec<Vec<Vec<i32>>> = lines[1..]
            .chunks(6)
            .map(|lines| {
                let square = lines[1..]
                    .iter()
                    .map(|line| {
                        line.split_ascii_whitespace()
                            .map(|n| n.parse::<i32>().unwrap())
                            .collect()
                    })
                    .collect::<Vec<Vec<i32>>>();
                (0..square[0].len())
                    .map(|index| square.iter().map(|v| v[index].clone()).collect())
                    .chain(square.clone().into_iter())
                    .collect()
            })
            .collect();
        (
            lines[0]
                .split(",")
                .map(|i| i.parse().unwrap())
                .collect::<Vec<i32>>(),
            grids,
        )
    };
    let numbers_by_index: HashMap<_, _> =
        numbers.iter().enumerate().map(|(x, y)| (*y, x)).collect();

    let grids_with_end: Vec<_> = grids
        .iter()
        .map(|grid| {
            (
                grid,
                grid.iter()
                    .map(|line| {
                        line.iter()
                            .map(|n| *numbers_by_index.get(n).unwrap_or(&usize::MAX))
                            .max()
                            .unwrap()
                    })
                    .min()
                    .unwrap(),
            )
        })
        .collect();

    let score = |grid: &Vec<Vec<i32>>, end| {
        (numbers[end]
            * grid[0..5]
                .concat()
                .iter()
                .filter(|i| numbers_by_index[i] > end)
                .cloned()
                .sum::<i32>()) as usize
    };

    let (best_grid, best_end) = grids_with_end.iter().min_by_key(|(_, i)| i).unwrap();
    let (worst_grid, worst_end) = grids_with_end.iter().max_by_key(|(_, i)| i).unwrap();

    (
        Some(score(best_grid, *best_end)),
        Some(score(worst_grid, *worst_end)),
    )
}

fn func5(input: &str) -> Output {
    struct Segment {
        start: (i32, i32),
        end: (i32, i32),
    }
    let re = Regex::new(r"^(\d+),(\d+) -> (\d+),(\d+)$").expect("Regex creation failed");
    let segments = input
        .lines()
        .map(|s| {
            let cap = re.captures(s).unwrap();
            Segment {
                start: (cap[1].parse().unwrap(), cap[2].parse().unwrap()),
                end: (cap[3].parse().unwrap(), cap[4].parse().unwrap()),
            }
        })
        .collect::<Vec<_>>();
    let bad_points = segments
        .iter()
        .filter(|s| s.start.0 == s.end.0 || s.start.1 == s.end.1)
        .flat_map(|&Segment { start, end }| {
            let (start, end) = (min(start, end), max(start, end));
            (start.0..end.0 + 1).flat_map(move |x| (start.1..end.1 + 1).map(move |y| (x, y)))
        })
        .fold(HashMap::new(), |mut counts, element| {
            *counts.entry(element).or_insert(0 as usize) += 1;
            counts
        })
        .iter()
        .filter(|(_, c)| **c > 1)
        .count();
    let very_bad_points = segments
        .iter()
        .flat_map(|&Segment { start, end }| {
            let direction = (end.0 - start.0, end.1 - start.1);
            let length = max(direction.0.abs(), direction.1.abs());
            (0..length + 1).map(move |x| {
                (
                    start.0 + direction.0 / length * x,
                    start.1 + direction.1 / length * x,
                )
            })
        })
        .fold(HashMap::new(), |mut counts, element| {
            *counts.entry(element).or_insert(0 as usize) += 1;
            counts
        })
        .iter()
        .filter(|(_, c)| **c > 1)
        .count();

    (Some(bad_points), Some(very_bad_points))
}

fn func6(input: &str) -> Output {
    let population: HashMap<i32, usize> = input
        .trim_end()
        .split(",")
        .map(|d| d.parse().unwrap())
        .fold(HashMap::new(), |mut map, p| {
            *map.entry(p).or_default() += 1;
            map
        });
    let iterate = |current_pop: HashMap<i32, usize>| {
        let mut next_pop = HashMap::new();
        current_pop.into_iter().for_each(|(age, size)| {
            if age == 0 {
                *next_pop.entry(6).or_default() += size;
                *next_pop.entry(8).or_default() += size;
            } else {
                *next_pop.entry(age - 1).or_default() += size;
            }
        });
        next_pop
    };
    let population80 = (0..80).fold(population.clone(), |pop, _| iterate(pop));
    let population256 = (0..256).fold(population.clone(), |pop, _| iterate(pop));
    (
        Some(population80.values().sum::<usize>()),
        Some(population256.values().sum::<usize>()),
    )
}

fn func7(input: &str) -> Output {
    let positions: HashMap<i32, usize> = input
        .trim_end()
        .split(",")
        .map(|c| c.parse::<i32>().unwrap())
        .fold(HashMap::new(), |mut map, c| {
            *map.entry(c).or_default() += 1;
            map
        });
    let sorted_positions = {
        let mut vec = positions.clone().into_iter().collect::<Vec<(i32, usize)>>();
        vec.sort_by_key(|(p, _)| *p);
        vec
    };
    let mut num_crabs = (positions.values().sum::<usize>() / 2) as i64;
    let &(position, _) = sorted_positions
        .iter()
        .skip_while(move |(_, crabs)| {
            num_crabs -= *crabs as i64;
            num_crabs > 0
        })
        .next()
        .unwrap();
    let distance = positions
        .iter()
        .map(|(p, n)| (p - position).abs() as usize * n)
        .sum();

    let crab_eng_distance = |position: i32| -> usize {
        positions
            .iter()
            .map(|(p, n)| {
                let distance = (position - p).abs() as usize;
                distance * (distance + 1) * n / 2
            })
            .sum()
    };
    let smallest_position = *positions.keys().min().unwrap();
    let largest_position = *positions.keys().max().unwrap();
    let best_crab = (smallest_position..largest_position)
        .map(crab_eng_distance)
        .min()
        .unwrap();
    (Some(distance), Some(best_crab))
}

fn func8(input: &str) -> Output {
    type Code = BTreeSet<char>; // We need to hash it
    struct Cipher {
        numbers: [Code; 10],
        output: [Code; 4],
    }
    let regex = Regex::new(r"^([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+) \| ([a-g]+) ([a-g]+) ([a-g]+) ([a-g]+)$").expect("Bad regex");
    let ciphers = input
        .lines()
        .map(|s| {
            let cap = regex.captures(s).expect("Failed to parse");
            let to_code = |s: &str| s.chars().collect::<BTreeSet<char>>();
            Cipher {
                numbers: [
                    to_code(&cap[1]),
                    to_code(&cap[2]),
                    to_code(&cap[3]),
                    to_code(&cap[4]),
                    to_code(&cap[5]),
                    to_code(&cap[6]),
                    to_code(&cap[7]),
                    to_code(&cap[8]),
                    to_code(&cap[9]),
                    to_code(&cap[10]),
                ],
                output: [
                    to_code(&cap[11]),
                    to_code(&cap[12]),
                    to_code(&cap[13]),
                    to_code(&cap[14]),
                ],
            }
        })
        .collect::<Vec<_>>();
    let easy_number_count = ciphers
        .iter()
        .map(|Cipher { numbers: _, output }| {
            output
                .iter()
                .filter(|code| [2, 3, 4, 7].contains(&code.len()))
                .count()
        })
        .sum::<usize>();
    type Decoder = HashMap<Code, i32>;
    let analyze = |code: &[Code]| -> Decoder {
        let mut decoder = Decoder::new();
        let one = code.iter().find(|c| c.len() == 2).expect("Can't find one");
        let four = code.iter().find(|c| c.len() == 4).expect("Can't find four");
        let seven = code
            .iter()
            .find(|c| c.len() == 3)
            .expect("Can't find seven");
        let eight = code
            .iter()
            .find(|c| c.len() == 7)
            .expect("Can't find eight");
        let three = code
            .iter()
            .find(|c| c.len() == 5 && c.is_superset(&one))
            .expect("Can't find three");
        let nine = code
            .iter()
            .find(|c| c.len() == 6 && c.is_superset(&three))
            .expect("Can't find nine");
        let zero = code
            .iter()
            .find(|c| c.len() == 6 && c != &nine && c.is_superset(&seven))
            .expect("Can't find zero");
        let six = code
            .iter()
            .find(|c| c.len() == 6 && c != &nine && c != &zero)
            .expect("Can't find six");
        let five = code
            .iter()
            .find(|c| c.len() == 5 && c.is_subset(&six))
            .expect("Can't find five");
        let two = code
            .iter()
            .find(|c| c.len() == 5 && c != &five && c != &three)
            .expect("Can't find two");
        decoder.insert(one.clone(), 1);
        decoder.insert(two.clone(), 2);
        decoder.insert(three.clone(), 3);
        decoder.insert(four.clone(), 4);
        decoder.insert(five.clone(), 5);
        decoder.insert(six.clone(), 6);
        decoder.insert(seven.clone(), 7);
        decoder.insert(eight.clone(), 8);
        decoder.insert(nine.clone(), 9);
        decoder.insert(zero.clone(), 0);
        decoder
    };
    let total = ciphers
        .iter()
        .map(|Cipher { numbers, output }| {
            let decoder = analyze(numbers);
            (1000 * decoder[&output[0]]
                + 100 * decoder[&output[1]]
                + 10 * decoder[&output[2]]
                + decoder[&output[3]]) as usize
        })
        .sum::<usize>();
    (Some(easy_number_count), Some(total))
}

fn func9(input: &str) -> Output {
    let map: Vec<Vec<u32>> = input
        .lines()
        .map(|l| l.chars().map(|c| c.to_digit(10).unwrap()).collect())
        .collect();
    let max_x = map.len() as i32;
    let max_y = map[0].len() as i32;
    let get_neighbors = |x, y| {
        let x = x as i32;
        let y = y as i32;
        vec![(x, y + 1), (x + 1, y), (x - 1, y), (x, y - 1)]
            .into_iter()
            .filter(|&(x, y)| x >= 0 && y >= 0 && x < max_x && y < max_y)
            .map(|(x, y)| (x as usize, y as usize))
    };
    let low_points = {
        let mut l = Vec::new();
        for (x, line) in map.iter().enumerate() {
            for (y, &cell) in line.iter().enumerate() {
                if get_neighbors(x, y).all(|(x, y)| cell < map[x][y]) {
                    l.push((x, y));
                }
            }
        }
        l
    };
    let bassins = {
        let mut point_to_bassin: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        for x in 0..map.len() {
            for y in 0..map[0].len() {
                point_to_bassin.insert((x, y), (x, y));
            }
        }
        fn find(
            x: usize,
            y: usize,
            map: &Vec<Vec<u32>>,
            bassins: &mut HashMap<(usize, usize), (usize, usize)>,
        ) -> (usize, usize) {
            if bassins[&(x, y)] != (x, y) {
                return bassins[&(x, y)];
            }
            let max_x = map.len() as i32;
            let max_y = map[0].len() as i32;
            let get_neighbors = |x, y| {
                let x = x as i32;
                let y = y as i32;
                vec![(x, y + 1), (x + 1, y), (x - 1, y), (x, y - 1)]
                    .into_iter()
                    .filter(|&(x, y)| x >= 0 && y >= 0 && x < max_x && y < max_y)
                    .map(|(x, y)| (x as usize, y as usize))
            };
            if let Some((nx, ny)) = get_neighbors(x, y).find(|&(nx, ny)| map[x][y] > map[nx][ny]) {
                let bassin_point = find(nx, ny, map, bassins);
                bassins.insert((x, y), bassin_point);
                bassin_point
            } else {
                (x, y)
            }
        }
        for (x, line) in map.iter().enumerate() {
            for (y, &cell) in line.iter().enumerate() {
                if cell == 9 {
                    continue;
                }
                find(x, y, &map, &mut point_to_bassin);
            }
        }
        point_to_bassin
    };
    let bassins_by_size: HashMap<(usize, usize), i32> = bassins
        .iter()
        .filter(|&(&(x, y), _)| map[x][y] != 9)
        .fold(HashMap::new(), |mut acc, (_, &bassin)| {
            *acc.entry(bassin).or_default() += 1;
            acc
        });
    let largest_bassins = {
        let mut bassin_sizes = bassins_by_size.values().collect::<Vec<_>>();
        bassin_sizes.sort();
        bassin_sizes.reverse();
        bassin_sizes.into_iter().take(3).product::<i32>()
    };

    (
        Some(low_points.iter().map(|&(x, y)| map[x][y] + 1).sum::<u32>() as usize),
        Some(largest_bassins as usize),
    )
}

fn func10(input: &str) -> Output {
    #[derive(PartialEq)]
    enum ParType {
        Curly,
        Round,
        Square,
        Angle,
    }
    enum ParseStatus {
        Ok(Vec<ParType>),
        Corrupted(ParType, ParType),
        Incomplete(Vec<ParType>),
    };
    let line = input.lines().collect::<Vec<_>>();
    fn parse(line: &str) -> ParseStatus {
        match line
            .chars()
            .fold(ParseStatus::Ok(Vec::new()), |status, char| {
                let mut stack = match status {
                    ParseStatus::Ok(s) => s,
                    _ => return status,
                };
                match char {
                    '(' => {
                        stack.push(ParType::Round);
                        ParseStatus::Ok(stack)
                    }
                    '{' => {
                        stack.push(ParType::Curly);
                        ParseStatus::Ok(stack)
                    }
                    '[' => {
                        stack.push(ParType::Square);
                        ParseStatus::Ok(stack)
                    }
                    '<' => {
                        stack.push(ParType::Angle);
                        ParseStatus::Ok(stack)
                    }
                    ')' => {
                        let popped = stack.pop().unwrap();
                        if popped != ParType::Round {
                            ParseStatus::Corrupted(popped, ParType::Round)
                        } else {
                            ParseStatus::Ok(stack)
                        }
                    }
                    '}' => {
                        let popped = stack.pop().unwrap();
                        if popped != ParType::Curly {
                            ParseStatus::Corrupted(popped, ParType::Curly)
                        } else {
                            ParseStatus::Ok(stack)
                        }
                    }
                    ']' => {
                        let popped = stack.pop().unwrap();
                        if popped != ParType::Square {
                            ParseStatus::Corrupted(popped, ParType::Square)
                        } else {
                            ParseStatus::Ok(stack)
                        }
                    }
                    '>' => {
                        let popped = stack.pop().unwrap();
                        if popped != ParType::Angle {
                            ParseStatus::Corrupted(popped, ParType::Angle)
                        } else {
                            ParseStatus::Ok(stack)
                        }
                    }
                    _ => panic!("lol wut"),
                }
            }) {
            ParseStatus::Ok(stack) if ! stack.is_empty() => {
                ParseStatus::Incomplete(stack)
            }
            status => status,
        }
    }
    let parse_status = line.iter().map(|s| parse(s)).collect::<Vec<_>>();
    let corrupted  = parse_status.iter().filter_map(|status| match status {
        ParseStatus::Ok(_) => None,
        ParseStatus::Corrupted(_, ParType::Square) => Some(57),
        ParseStatus::Corrupted(_, ParType::Curly) => Some(1197),
        ParseStatus::Corrupted(_, ParType::Round) => Some(3),
        ParseStatus::Corrupted(_, ParType::Angle) => Some(25137),
        ParseStatus::Incomplete(_) => None,
    }).sum::<usize>();
    let mut incomplete= parse_status.iter().filter_map(|status| match status {
        ParseStatus::Ok(_) => None,
        ParseStatus::Corrupted(_, _) => None,
        ParseStatus::Incomplete(stack) => {
            Some(stack.iter().rev().fold(0 as usize, |acc, par| match par {
                ParType::Curly => 5 * acc + 3,
                ParType::Round => 5 * acc + 1,
                ParType::Square => 5 * acc + 2,
                ParType::Angle => 5 * acc + 4,
            }))
        },
    }).collect::<Vec<_>>();
    incomplete.sort();
    (Some(corrupted), Some(incomplete[incomplete.len() / 2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn problem1() {
        let input = "1
        2
        3
        4
        1
        2
        1
        2"
        .replace(" ", "");
        assert_eq!(func1(&input), (Some(5), Some(2)));
    }

    #[test]
    fn problem2() {
        let input = "forward 5\ndown 5\nforward 8\nup 3\ndown 8\nforward 2";
        assert_eq!(func2(input), (Some(150), Some(900)));
    }

    #[test]
    fn problem3() {
        let input = "00100
        11110
        10110
        10111
        10101
        01111
        00111
        11100
        10000
        11001
        00010
        01010"
            .replace(" ", "");
        assert_eq!(func3(&input), (Some(198), Some(230)));
    }

    #[test]
    fn problem4() {
        let input = "7,4,9,5,11,17,23,2,0,14,21,24,10,16,13,6,15,25,12,22,18,20,8,19,3,26,1

22 13 17 11  0
 8  2 23  4 24
21  9 14 16  7
 6 10  3 18  5
 1 12 20 15 19

 3 15  0  2 22
 9 18 13 17  5
19  8  7 25 23
20 11 10 24  4
14 21 16 12  6

14 21 17 24  4
10 16 15  9 19
18  8 23 26 20
22 11 13  6  5
 2  0 12  3  7";
        assert_eq!(func4(input), (Some(4512), Some(1924)));
    }
    #[test]
    fn problem5() {
        let input = "0,9 -> 5,9
8,0 -> 0,8
9,4 -> 3,4
2,2 -> 2,1
7,0 -> 7,4
6,4 -> 2,0
0,9 -> 2,9
3,4 -> 1,4
0,0 -> 8,8
5,5 -> 8,2";
        assert_eq!(func5(input), (Some(5), Some(12)));
    }

    #[test]
    fn problem6() {
        let input = "3,4,3,1,2";
        assert_eq!(func6(input), (Some(5934), Some(26984457539)));
    }
    #[test]
    fn problem7() {
        let input = "16,1,2,0,4,2,7,1,2,14";
        assert_eq!(func7(input), (Some(37), Some(168)));
    }

    #[test]
    fn problem8() {
        let input =
            "be cfbegad cbdgef fgaecd cgeb fdcge agebfd fecdb fabcd edb | fdgacbe cefdb cefbgd gcbe
edbfga begcd cbg gc gcadebf fbgde acbgfd abcde gfcbed gfec | fcgedb cgb dgebacf gc
fgaebd cg bdaec gdafb agbcfd gdcbef bgcad gfac gcb cdgabef | cg cg fdcagb cbg
fbegcd cbd adcefb dageb afcb bc aefdc ecdab fgdeca fcdbega | efabcd cedba gadfec cb
aecbfdg fbg gf bafeg dbefa fcge gcbea fcaegb dgceab fcbdga | gecf egdcabf bgf bfgea
fgeab ca afcebg bdacfeg cfaedg gcfdb baec bfadeg bafgc acf | gebdcfa ecba ca fadegcb
dbcfg fgd bdegcaf fgec aegbdf ecdfab fbedc dacgb gdcebf gf | cefg dcbef fcge gbcadfe
bdfegc cbegaf gecbf dfcage bdacg ed bedf ced adcbefg gebcd | ed bcgafe cdgba cbgef
egadfb cdbfeg cegd fecab cgb gbdefca cg fgcdab egfdb bfceg | gbdfcae bgc cg cgb
gcafb gcf dcaebfg ecagb gf abcdeg gaef cafbge fdbac fegbdc | fgae cfgab fg bagce";
        assert_eq!(func8(input), (Some(26), Some(61229)));
    }

    #[test]
    fn problem9() {
        let input = "2199943210
3987894921
9856789892
8767896789
9899965678";
        assert_eq!(func9(input), (Some(15), Some(1134)));
    }
    #[test]
    fn problem10() {
        let input = "[({(<(())[]>[[{[]{<()<>>
[(()[<>])]({[<{<<[]>>(
{([(<{}[<>[]}>{[]{[(<()>
(((({<>}<{<{<>}{[]{[]{}
[[<[([]))<([[{}[[()]]]
[{[{({}]{}}([{[{{{}}([]
{<[[]]>}<{[{[{[]{()[[[]
[<(<(<(<{}))><([]([]()
<{([([[(<>()){}]>(<<{{
<{([{{}}[<[[[<>{}]]]>[]]";
        assert_eq!(func10(input), (Some(26397), Some(288957)));
    }
}
