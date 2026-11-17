# Training notes

Last updated: 17 November 2026 (shortened again; the long version with every check and interim reading is
in git, commit `00a719e`, and before that `7c4277a`).
This file is the memory of the project's training decisions. It is kept up to date whenever something
that influences the training changes.

## Where things stand

- **Size is not the limit.** Large (1.84M weights) ended level with medium (0.67M): 52.3% at 500
  simulations, 50.0% at 2000.
- **Training settings alone do not fix it.** Twice the games per round gave +2.4 points, twice the
  simulations nothing.
- **Line layers (design D5):** learn about twice as fast per lesson and lead large in the training checks
  against medium (56.0% against 51.1% over rounds 124 to 160), but **lose a direct match against large**
  (46.9% / 47.7%) and cost 1.4 to 1.5 times the time per round. Not stronger; not continued for now.
- **Lessons about the long game** (owner, result as three chances, move values): 0 to 3 points, inside the
  chance between two equal runs. Not used.
- **Small network with many games (`small_many`):** about level with large per hour of training, far behind
  per game, and it levels off lower. Stopped at round 105 by the stop rule. A small network does not beat large.
- **The one lever that helped is more games**, and only where the network is big enough. Everything else
  ended at about the same level. Plan: put the remaining compute into large with many more games (see
  "Proposal for the final model").

## Plan for the remaining units (the user's plan, 17 November)

99 units left: about 4 for the rest of `small_many`, about 10 kept for later tests, about 85 for **one final
model, as good as possible and making few mistakes**. Before that: a thorough review of all experiments, with
5 to 10 units of tests only if needed. The user's goal and doubt: a 50% match where both sides win and lose
on the same openings means both made mistakes; the final model should, against medium on the same openings,
lose rarely and hold the draw.

Checked on the saved matches (both games of each opening, one from each side): large 240 against medium 240
at 2000 simulations, 118 openings: both games tied 68, one game decided and one tied 43, each side won once
(the side moving first both times) 4, large won both 2, medium won both 1. At 500 simulations: 55 / 54 / 5 /
2 / 2. So "each wins one" is rare; the typical mistake is one decided game in a pair, on about 40% of the
openings at 2000 simulations (from 61% for large round 40). That share is a better measure of mistakes than
the score.

**Measure for the final model (the user's rule):** per opening, add the points of both games. Below 1 (for
example a loss and a tie, or two losses) = played worse on that opening; above 1 = played better; 1 = equal.
Openings worse / equal / better (118 openings; from the saved match files):

| | 500 simulations | 2000 simulations |
|---|---|---|
| large 40 against medium 240 | 88 / 26 / 4 | 81 / 32 / 5 |
| large 120 against medium 240 | 56 / 49 / 13 | 52 / 50 / 16 |
| large 240 against medium 240 | 23 / 60 / 35 | 22 / 72 / 24 |
| line network 160 against medium 240 | 19 / 53 / 46 | 26 / 59 / 33 |
| line network 160 against large 240 | 36 / 54 / 28 | 27 / 73 / 18 |

Goal: worse on few openings (large 240 is still worse on about 22). Note that a few openings may favour one
side so much that "equal" there means both played well (each won once with the first move: 4 at 2000).

## Proposal for the final model (17 November; tests built, final run not decided yet)

**Reading of all experiments together.** Medium and large ended level at the same games; the line network
learned faster but ended level and lost to large directly; the lessons gave 0 to 3 points; 800 simulations
gave nothing; more games per round (fresh data per learning step) gave +2.4 even at the lowest learning rate;
the small network with many games was capped by its size. So the networks are limited by the number of games
(about 123,000 for large; AlphaZero chess used 44 million), not by size or design, as long as the network is
not too small. The learning-rate steps gave +5 to 6 points each time but less at the last step.

**Proposal (about 85 units):**
1. Tests, about 7 units, two Colab sessions at once, each continuing large from round 240 like `more_games`
   (1024 games per round, each position used twice) for the same training time as `more_games`' 20 rounds
   at today's speed (about 1.6 hours), then a direct match against `more_games`' round 260:
   - `fast_moves`: KataGo's trick, a full search (400) on a quarter of the moves, 100 simulations on the
     others; only the full ones are learned from. Guessed 2.5 times the games per hour.
   - `sims_200`: 200 simulations on every move; guessed about 1.7 times the games per hour.
   - Plus the speed of large with the search on all cores (512 and 1024 games at once).
   - Rule: a test is used for the final run only if it wins the direct match against `more_games` 260 by
     53% or more (236 games, 500 simulations); otherwise 400 simulations on every move as in `more_games`.
2. Final run, about 75 units (about 39 hours): large continued from the best of these networks; 1024 games
   per round; search on all cores; learning rate 0.0001, 0.00003 for the last 15%; memory 1.5 million
   positions (the network changes slowly at this rate, so older games stay good lessons; guessed); checks
   against large 240 (512 games, per-opening worse / equal / better) every 8 rounds; snapshot every 8 rounds.
   Without the trick about 550,000 games (4.5 times large's run), with it perhaps twice that (guessed).
3. End, about 3 units: direct matches of the last snapshots and of their average (weight averaging) against
   large 240 and medium 240 at 500 and 2000 simulations, per-opening counts; the best is the final model.

**Built (17 November, the user said yes):**
- `FULL_SEARCH_SHARE` and `FAST_SIMULATIONS` in `src/train.py` (KataGo's trick; the choice is made per move for
  all games of an iteration at once; fast moves have no noise and are not learned from). `TIME_LIMIT` stops a
  training by computing time (new log column `work_seconds`, self-play and learning without the checks).
- Tests `fast_moves` and `sims_200` in `EXPERIMENTS`, time limit 5,600 s after iteration 240 (= 20 x 280 s,
  more_games' 20 iterations at today's speed, guessed; if the true time is lower the tests get a little more).
- `python3 src/train.py match <test> <other test>`: the newest networks of two tests, 128 openings of
  test_tournament.py from both sides, 500 simulations, per-opening worse / equal / better; saved in
  `experiments/<test>/match_<other>.csv`.
- Notebook `colab_final_tests.ipynb` (one session per test, `TEST` in the first cell); the `fast_moves` session
  also measures large's speed on 1 core and all cores at 512 and 1024 games at once.
- Tried on the laptop with a tiny setup (small network, 16 to 32 games): 24% of the moves got the full search,
  only those positions were kept (201 of about 900); fast moves had no noise; the test stopped at its time
  limit and did not go on after a restart; `sims_200` changed the simulations; the match was saved and a second
  call only printed it. Not run on the L4.

Not proposed, and why: bigger network (size is not the limit yet); line layers (lost to large, 1.45 times
slower); medium (2.3 times faster, but less room to grow over a long run; small showed the size cap);
long-game lessons (no gain); more simulations (no gain); proofs of game ends in the search (traps are rare;
moves towards minimax); changes to the randomness of self-play (untested, risky in the final run).

## `small_many` (17 November, Colab L4, stopped at round 105)

- Small size (32 channels, 4 blocks, 0.19M weights) from scratch; 2048 games per round, all at once; search
  tree on all processor cores. Settings in `RUN_SETTINGS` in `src/train.py`; notebook `colab_small_many.ipynb`;
  command `python -u src/train.py run small_many 150`.
- Schedule by games, like large: 200 simulations until round 25 (51,200 games = large's round 100), then 400;
  learning rate 0.001, 0.0003 from round 76, 0.0001 from 126 (guessed). Memory 450,000 positions, 900,000
  from round 76; every position used 4 times. 150 rounds = 307,000 games, 2.5 times large's whole run.
- Judged by: checks against medium every 4 rounds, compared with large **at the same number of games**
  (large: 48.4% at 51,200, 49.0% at 61,400, 50.8% at 81,900, 54.7% at 96,300, 53.9% at 122,900 = round 240);
  and at the end a direct match against medium and large (last notebook cell,
  `alphazero_small_many_tournament.csv`).
- Measured (log `models/alphazero_small_many_log.csv`): 83 s per round at 200 simulations, 139 s at 400
  (the guess was 100 to 110). Rounds 1 to 60 took 1.92 hours of training time. Guessed for all 150: about
  5.4 hours, about 11 units.
- **Interim at round 60 (122,880 games), points against medium:**

  | Round | 12 | 24 | 36 | 44 | 48 | 52 | 56 | 60 |
  |---|---|---|---|---|---|---|---|---|
  | hours | 0.29 | 0.55 | 1.00 | 1.31 | 1.46 | 1.61 | 1.77 | 1.92 |
  | `small_many` | 20.5% | 32.2% | 34.0% | 36.3% | 36.7% | 38.7% | 41.8% | 36.5% |
  | large at the same games | 35.5% | 47.3% | 53.5% | 50.0% | 52.1% | 53.5% | 53.7% | 53.9% |
  | `large_repeat` at the same hours | 3.3% | 12.5% | 29.5% | 29.1% | 34.8% | 34.0% | 39.8% | 40.2% |

  - **By games it is far behind:** 13 to 17 points below large at the same number of games. The small network
    learns much less from each game, as expected.
  - **By hours it was ahead early and is now level:** 15 to 20 points ahead of `large_repeat` in the first
    half hour, about 5 points ahead around one hour, level by 1.8 to 1.9 hours (mean of the last 3 checks 39.0%
    against 38.0%). So far the extra games make up for the smaller network, but no more than that.
  - Still rising slowly (about 1 point per 4 rounds over rounds 24 to 60). The learning-rate step at round 76 is
    still to come; it lifted large by about 5 points. Policy loss 1.28 at round 60 (large 1.29 at round 100,
    same simulations).
  - To beat large it has to reach about 55% by the end; at today's slope that needs the step to help clearly
    (guessed).
- **Stop rule, set at round 60:** decide at round 100 (6 checks after the learning-rate step at 76). Mean of
  rounds 80 to 100 below 48% (large at its round 100: 48.4%): stop, the many games do not beat large. 50% or
  more: run to 150. In between: run to 124 and decide again with the same 50%. Stopping at 100 saves about
  2 hours and 4 units.
- **Result (round 105, 215,000 games, 3.69 hours of training time):**

  | Round | 64 | 68 | 72 | 76 | 80 | 84 | 88 | 92 | 96 | 100 | 104 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | `small_many` | 41.4% | 38.3% | 39.6% | 43.0% | 41.4% | 44.5% | 45.3% | 41.2% | 42.6% | 47.1% | 45.1% |

  - Mean of rounds 80 to 100: **43.7%, below the 48% of the rule: stopped.** The learning-rate step at 76 lifted
    it by about 3 points (40.6% over rounds 64 to 76), less than large's 5.
  - By hours: large at the same training time (about round 120 at today's speed, guessed) stood at about 47 to
    49%, so small is level or slightly behind. By games: large reached 53.9% with 123,000 games; small has 43
    to 47% with 215,000. Value loss about 0.114 at round 104 (large: 0.11 to 0.12 around rounds 100 to 120 at the same
    simulations), policy loss 1.23 (large 1.25 at round 120).
  - No direct match played (not needed for the decision; saves about 0.5 units).

## Settings of the large run (`src/train.py`, the reference for all comparisons)

| Round | Time from start | What happens |
|---|---|---|
| 1 to 100 | 0 to 3.9 hours | 200 simulations per move, learning rate 0.001, memory 300,000 positions |
| 101 | 3.9 hours | 400 simulations; rounds take about twice as long |
| 121 | 5.5 hours | Learning rate 0.0003 |
| 191 | about 10 hours | Learning rate 0.0001, memory 600,000 |
| 240 | about 14 hours | Stops (before the faster copy; now about 1.5 times as fast) |

Also: 512 games per round, all at once; every position used 4 times; snapshot every 4 rounds; every 4 rounds
256 games against medium (100 simulations, same openings), uncertain by about 3 points each.

## Decisions and why

- **Pure network.** No minimax in play or training; minimax is only a measuring opponent (the user's preference).
- **From scratch.** A large network taught only from medium's last games scored 19% against medium.
- **Judge by games, not by the loss.** Medium's loss fell from round 61 to 130 while its strength stood still;
  design C2 had the lowest loss but was not the strongest; the line network had lower losses than large and
  still lost the direct match.
- **Judge by a direct match, not only by the checks against medium.** The line network led in the checks
  and lost to large directly.
- **Chance between two equal trainings is about 3 points** (`large_repeat`), so only a lead of 4 points or
  more over many checks counts. Set the rule before the run.
- **Learning-rate rule.** Keep the rate while the checks rise; lower it when they are flat. Every step so far
  lifted the checks by about 5 to 6 points.
- **Search setting (c_puct) stays 1.5.** A narrower search helps only at short thinking time.
- **Change one thing at a time**; compare losses only at the same number of simulations.

## What was tried and what came out

### Bigger network: large against medium

Training checks (256 games, 100 simulations):

| Rounds | 20 | 40 | 60 | 80 | 100 | 124 to 160 | 164 to 188 | 192 to 216 | 220 to 240 |
|---|---|---|---|---|---|---|---|---|---|
| Points | 13.7% | 28.5% | 35.4% | 42.8% | 48.4% | 51.1% | 52.3% | 53.2% | 54.8% |

Snapshot match against medium round 240:

| Large round | 40 | 80 | 120 | 160 | 200 | 240 |
|---|---|---|---|---|---|---|
| 500 simulations | 25.8% | 40.6% | 37.7% | 48.4% | 52.1% | 52.3% |
| 2000 simulations | 28.9% | 41.4% | 39.1% | 48.0% | 52.0% | 50.0% |

The last 40 rounds added nothing measurable. The checks at 100 simulations overstate large by 2 to 4
points. Ties grow with thinking time (192 of 256 at 2000). Moving first scores about 57%.

### Training settings (large continued from round 240, checks with 512 games)

Baseline 52.7%. `more_games` (1024 games per round, each position used 2 times): 55.1%, +2.4 points, likely
real. `more_simulations` (800): 52.8%, nothing. Both at the lowest learning rate.

### Search, medium and uttt.ai

- **Medium** levelled off from about round 140. Against minimax (2000 simulations): 96/88/88/76% against
  depth 12/13/14/15; 4000 simulations added nothing.
  At 2000 simulations against depth 15: 108 wins, 51 ties, 17 losses of 176 (`models/alphazero_medium_vs_minimax.csv`;
  the moves of those games were not saved).
- **Endgame check** (`src/test_endgame.py`, 12 November; `models/alphazero_medium_endgame.csv`): medium round
  132 against **itself**, 64 games, 2000 simulations; minimax depth 14 checked 2075 network moves from move 20.
  Missed proven win: 7 of 109 moves where a win was proven (some may only be slower wins). Moved into a proven
  loss when nothing was proven before: 8 of 1880. Games: 48 ties, 16 decisive. So traps are rare
  but not absent. Not run against minimax (the script can: `OPPONENT_DEPTH = 15`).
- **Search setting** against 1.5: 0.5 scored 55.1% at 500 and 49.4% at 2000; 0.75 54.7% / 50.0%; 2.5 42.4%.
- **Search depth:** the main line grows by 2 to 3 moves per doubling of simulations (12 at 500, 17 at 2000).
- **Against uttt.ai's network** (236 games, our search for both): 47.0% at 500, 50.6% at 2000, so level.
  Their network is 2.2 times slower per position. (Files local only in `external/`; results in
  `models/alphazero_medium_vs_utttai.csv`.)

### Network designs (`src/test_designs.py`)

**Method.** Each design learns from large's saved positions (6,000 steps, medium size), then is measured on
held-back positions (distance to the search's moves, value error) and in a 236-game match at 200
simulations (uncertain by about 2.5 points; two runs of the same design differ by about 5). Shows how well a
design learns from given lessons, not how it trains itself.

| Design | What is changed | Distance | Value error | Against D0 | Against D5 |
|---|---|---|---|---|---|
| D0 | today's design | 0.285, 0.276 | 0.136, 0.133 | 46.0% | 28.6% |
| D1 | filters stay inside the small board | 0.246 | 0.130 | 59.7%, 56.6% | |
| D2 | learned small-board summary | 0.273 | 0.132 | 48.1% | |
| D3 | permanent board track | 0.283 | 0.134 | 53.8% | |
| D4 | each board hears from the 9 cells sending into it | 0.267 | 0.133 | 55.5% | |
| **D5** | **line layers** | **0.232, 0.235** | **0.128, 0.121** | **70.6%, 71.0%** | |
| D6 | exact facts as 8 extra input pictures | 0.268 | 0.134 | 61.2%, 56.4% | |
| D7 | whole body on the 3x3 big board (as uttt.ai) | 0.317 | 0.146 | 41.3% | 22.0% |
| C1 | D1 + D5 | 0.241, 0.235 | 0.122, 0.122 | 69.1%, 65.9% | 44.5%, 43.0% |
| C2 | D1 + D4 + D5 + D6 | 0.229, 0.230 | 0.120, 0.119 | 66.7%, 66.1% | 50.8%, 45.1% |

Night tests (laptop, 6.1 hours) against D5: D8 whole-board summary 39.8% / 41.1% (and 45.3% / 36.2% at large
size), 16 or 64 numbers per line instead of 32 and a learned board summary all below 50%. At large size D5
beat D0 66%; with 6 times longer learning the lead shrank to 61% (D5 after 18,000 steps = D0 after 36,000).
**So the line layers reach the same quality with about half the lessons, and the lead shrinks the longer both
learn.**

**How the line layers work.** In every block each cell is shortened to 32 numbers, added up along each of
the 72 small lines, worked on by one learned step shared by all lines, and each cell gets back the sum of the
lines through it; the same for the 8 lines of the big board. 7% more weights. They gain most where the whole
board matters (free choice of board, late game: value error -24%) and nothing in the opening. Hardest for
every network: the middle game with 3 to 4 boards decided.

### Line network trained itself (`large_lines`, 1.98M weights, large's settings)

| Round | 20 | 40 | 56 | 72 | 100 | 120 | 136 | 144 | 160 | mean 124 to 160 |
|---|---|---|---|---|---|---|---|---|---|---|
| line network | 18.8% | 40.0% | 44.5% | 42.2% | 46.7% | 52.1% | 55.1% | 59.0% | 57.4% | 56.0% |
| large | 13.7% | 28.5% | 33.6% | 43.9% | 48.4% | 49.0% | 48.8% | 53.5% | 50.8% | 51.1% |

- Rounds 1 to 100: ahead by up to 11 points until round 68, then flat at 41 to 49% and level at round 100.
  By hours no gain (7.9 against 3.9 hours, before its faster copy).
- After the learning-rate step it led in all 9 checks from round 124, by 4.9 points; that met the rule set
  beforehand (55% or more, lead of 4 or more). Rounds 101 to 160: 4.83 hours, 9.4 units. Round 132 is
  missing from the log.
- **Direct match** (round 160, 256 games per pair, `models/alphazero_large_lines_tournament.csv`):

  | | 500 simulations | 2000 simulations |
  |---|---|---|
  | against large (round 240) | 46.9% (39-162-55) | 47.7% (23-198-35) |
  | against medium (round 240) | 56.4% (63-163-30) | 51.2% (42-178-36) |
  | for comparison: large against medium | 52.3% | 50.0% |

  Not stronger than large; the lead against medium does not carry over to large (each number about 2 points
  uncertain; large also had 80 more rounds). Not tried: against large's snapshot of round 160 (equal rounds).

### Lessons about the long game (`large_long`, 100 rounds each, 16 November)

Three extra outputs used only while learning (3,600 weights, 3% more time): `owner` (who holds each open
small board at the end), `outcome` (win / tie / loss as three chances, from the result alone), `moves` (the
search's value of every tried move). Code: `LESSONS` in `src/Tic_Tac_Toe_NN.py`, `LESSON_WEIGHTS` and
`lesson_losses` in `src/train.py`, `root_move_values` in `src/MCTS.py`.

| | large (older run) | `large_repeat` | `large_long` | `large_long_no_owner` |
|---|---|---|---|---|
| Round 100 | 48.4% | 40.8% | 43.8% | 42.6% |
| mean of all 25 checks | 30.6% | 27.2% | 30.4% | 30.0% |
| mean of rounds 64 to 100 | 44.2% | 39.4% | 42.4% | 42.0% |

- `large_repeat` (large again, nothing changed) shows **chance between equal runs of about 3 points**.
- The lessons: +3 over `large_repeat`, level with the older large run, so 0 to 3 points. The owner of a
  board is hardly predictable (loss 0.91 against 1.10 for guessing) and dropping it changes nothing.

### Speed (Colab L4 unless said otherwise)

- **Faster network copy** (`FAST_PLAY`, `COMPILE` in `src/Tic_Tac_Toe_NN.py`): normalisation folded into the
  filters, other memory layout, half precision, `torch.compile`; checked against the real network after every
  change. Large self-play 7.1 to 4.3 ms per move (1.65 times); covers the line network too (6.8 ms; with
  `FAST_LINES` 5.9 to 6.2, kept off).
- **Card test (large, 512 at once):** the network itself is 86% of a question, copying 11%. So own code has
  at most about 10% left; more games at once cannot help large. Only less computing per position helps.
- **Search tree on all processor cores** (`SEARCH_THREADS = 'all'` in `src/MCTS.py`; numba `prange`, every
  game its own scratch space; exactly the same results as one core).
- **ms per move of a game, 400 simulations, faster copy:**

  | Games at once | 512 | 1024 | 2048 | 4096 |
  |---|---|---|---|---|
  | large, 1 core | 4.3 | 4.5 | 5.5 | |
  | medium, 12 cores | 2.9 | **1.9** | 2.0 | 2.4 |
  | small, 1 core | 3.0 | 2.2 | 2.0 | 2.2 |
  | small, 12 cores | 2.3 | 1.4 | **0.9** | 1.0 |

  Small is 4.8 times and medium 2.3 times as fast as large. For small at 2048 the card is busy only 40%;
  the limit is the processor preparing the questions on one core (next possible speed-up, guessed).
- The line network's slowdown in its first 100 rounds (217 s to 275 s per round) is unexplained; not longer
  games, not half precision.

## Known problems

- **Half precision fails in rare positions** (once ruined a network in round 33). Such positions are asked
  again in full precision (counted in the training line), damaged positions never reach the memory, and
  learning stops without saving if the loss is not a number.
- **The stop button on Colab can send its signal twice.** Since 17 November only the first counts; still wait
  for "Saved to ...".
- **A dropped session** loses at most the running round; the memory can be up to 4 rounds older.
- **Signs of trouble:** a large round over 150 s at 200 simulations; "not a number"; checks falling over
  several rounds.

## How to

- **Train on Colab:** `colab_train.ipynb` (or the run's own notebook), L4, run all cells; after a disconnect
  run all again. A Drive copy of a notebook does not update from git: open it from GitHub or add cells by hand.
  Colab allows two sessions at once.
- **Train another network with the same settings:** `python -u src/train.py run <network> <rounds>`
  (per-network settings in `RUN_SETTINGS`).
- **Go back to a snapshot:** `python -u src/train.py restore <round>`.
- **Training test:** `python3 src/train.py experiment <name>` (settings: `EXPERIMENTS`).
- **Matches:** `test_tournament.py`, settings set via `python -c` (see the last cell of a run's notebook);
  points are from the first-named player's side. Snapshots against medium: `colab_test.ipynb`.
- **Speed:** `python -u src/pretest.py bottleneck <network>` and `... card <network>`.
- **Design tests:** `src/test_designs.py` with `speed`, `train`, `match`, `summary`, `night`
  (`size=large`, `steps=36000`); results `models/design_tests.csv`, `models/design_matches.csv`.
- **Chart of a log:** `/usr/bin/python3 src/plot_log.py <network>`.
- **Files on Drive** (`Super-TikTakToe-models`): networks, `snapshots/`, logs, memories.
- **When a run ends:** Runtime > Disconnect and delete runtime. L4 costs 1.94 units per hour; units are not
  tight (state the cost, don't cut useful tests).
- **On the laptop** only `/usr/bin/python3` has the packages; long runs with `caffeinate -i`.

## What others have built (web research, 15 November 2026)

- **uttt.ai:** about 5M weights, 256 channels, 3 blocks on the 3x3 grid; two stages of about 8M positions;
  learns a value for every move.
- **CodinGame arena:** tiny networks (about 215,000 weights), fast search, over 300M positions.
- Only a rule variant is proven a first-player win; our rules are unsolved.
- Large saw about 6.4M positions; others used similar or far more data.

## Open and untested ideas

- Line network round 160 against large's snapshot of round 160 (equal rounds).
- Full search on only some moves (KataGo's trick; guessed up to 2 times faster).
- Preparing the network's questions faster on the processor (small's limit).
- Large with the search on all cores (guessed about 15% faster); TensorRT (guessed 1.2 to 1.5 times on the
  network part; not recommended).
- 1024 games per round from the start for large; large against minimax; D7 at equal thinking time.
- A bigger network (`xlarge`, 128/10) only once something makes size pay off: measure cost first, use the
  learning-rate rule, check against large, stop if not ahead by round 40 to 60.
- A learned description for each of the 19,683 fillings of a small board; an attention network with the
  game's links built in; games that carry over between rounds; tuning c_puct 1.0 to 1.5 in training,
  weight decay, and the value for untried moves.
