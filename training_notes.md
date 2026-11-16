# Training notes

Last updated: 16 November 2026 (shortened; the long version with every single check is in git, commit `7c4277a`).
This file is the memory of the project's training decisions. It is kept up to date whenever something
that influences the training changes.

## Where things stand

- **Large is finished and level with medium.** Large (96 channels, 8 blocks, 1.84 million weights, 240 rounds
  from scratch on Colab) scores 52.3% against medium at 500 simulations and 50.0% at 2000. So 2.75 times
  the weights gave almost nothing: size is not the limit. No bigger network for now.
- **Training settings alone do not fix it.** Continuing large with twice the games per round gave +2.4
  points, with twice the simulations nothing.
- **The line layers (design D5) learn about twice as fast per game, but so far do not end higher.** In
  self-play they were up to 11 points ahead of large at the same round, then stopped rising at round 48;
  at round 100 both stand at about 47%. On Colab a round took twice as long, so per hour nothing was won.

## Results of the night of 15 to 16 November 2026

**Colab: line network (`large_lines`, 1.98M weights) from scratch, 100 rounds with large's settings.**
Log: `models/alphazero_large_lines_log.csv`. Checks against medium (256 games, uncertain by about 3 points each):

| Round | Line network | Large | | Round | Line network | Large |
|---|---|---|---|---|---|---|
| 4 | 3.5% | 1.6% | | 56 | 44.5% | 33.6% |
| 8 | 8.4% | 4.7% | | 64 | 43.2% | 37.3% |
| 12 | 13.5% | 6.6% | | 72 | 42.2% | 43.9% |
| 20 | 18.8% | 13.7% | | 80 | 47.3% | 42.8% |
| 32 | 26.2% | 21.9% | | 88 | 42.8% | 45.7% |
| 40 | 40.0% | 28.5% | | 96 | 49.2% | 47.3% |
| 48 | 43.8% | 35.5% | | 100 | 46.7% | 48.4% |

- Ahead at every check up to round 68 (largest lead 11 points at rounds 40 and 56). From round 48 on it
  moves between 41% and 49% without rising; large catches up at round 72. Mean of the last 5 checks:
  47.3% against 46.4%, no difference.
- Its losses stay lower to the end: value loss 0.106 against 0.122, policy loss 1.282 against 1.293 at round 100.
  So it fits its own games better, but that did not become more points against medium.
- Time: 7.9 hours against large's 3.9 hours for the same 100 rounds (rounds went from 190 s to 300 s as the
  games got longer), about 15 units. Twice as long, not the guessed 20 to 50%. The faster network copy does
  not cover this design yet.
- By time: the line network reached 40% after 3.0 hours, large after about 2.8 hours. No gain per hour.
- Not measured: a direct match of the two networks at round 100 (the file `alphazero_large_lines.pt` is on
  Drive, not on the laptop). The check against medium is an indirect measure and may hide a difference.
- Guessed, not measured: both stop at the same level because the first 100 rounds' settings (200 simulations)
  set the limit there, not the design. Large only went on rising after round 100 with 400 simulations.

**Laptop: `src/test_designs.py night` (6.1 hours; log `models/pretest/night.log`).**

| Question | Result |
|---|---|
| Does anything add to the lines? (medium, 6,000 steps, match against D5, two runs) | No. D8 whole-board summary 39.8% / 41.1%; D5n 16 numbers per line 46.8% / 40.3%; D5w 64 numbers 45.1% / 44.7%; C3 with learned board summary 42.2% / 43.0%. All below 50%; 32 numbers per line stays. |
| Does the gain hold at large size? (6,000 steps, match against D0) | Yes. D5 66.1% / 65.7%; a second D0 gets 52.5% (the noise). Distance 0.240 against 0.265 / 0.254, value error 0.122 against 0.133. |
| Does the gap close with longer learning? (medium, 36,000 steps) | Partly. Match 61.4% (was 71% at 6,000 steps). Distance 0.182 against 0.201, value error 0.109 against 0.115. D5 after 18,000 steps is where D0 is after 36,000. |
| D8 at large size (match against D5) | Worse: 45.3% / 36.2%, distance 0.267 / 0.278 against 0.240. Dropped. |

**Read together:** the line layers reach the same quality with about half the lessons, at every size tested,
and the lead gets smaller the longer both learn. No test so far shows a higher final level.

## Lessons about the long game (prepared and first two runs, 16 November 2026)

Why: large and medium ended level, the line network ended level with large, so the limit is probably in the
lessons and not in the network. Today a position teaches the moves and one mixed number for the outcome.

**The three extra lessons** (outputs next to the two heads; used only while learning, play is unchanged and
no slower; 3,600 more weights):

| Lesson | What the network has to answer | Where it comes from |
|---|---|---|
| `owner` | For every small board that is still open: who holds it at the end, me / the other / nobody | the final board of the game |
| `outcome` | How the game ends, as three chances: win / tie / loss | the result alone (not mixed with the search value) |
| `moves` | How good the search found every move it tried | the search tree at the root; a move counts fully from 16 visits |

- The owner is not a plan and is not used to choose moves. It is the chance over the games that follow from
  this position, exactly like the value. If it depends too much on the opponent's later moves to be worth
  anything, the test without it shows that (the user's doubt, 16 November).
- Weights next to policy and value loss (`LESSON_WEIGHTS` in `src/train.py`): owner 0.5, outcome 0.5,
  moves 1.0. Guessed, not tuned.
- Code: outputs and `LESSONS` in `src/Tic_Tac_Toe_NN.py`; recording in `self_play`, memory (`EXTRA`) and
  `lesson_losses` in `src/train.py`; `root_move_values` in `src/MCTS.py`. Self-play always records the
  lessons; only networks with the outputs learn from them. Old memories load as "not known".
- Checked on the laptop with a small network: 166,994 owner answers against boards decided later in the same
  game, none wrong; the result class fits the result; the visit-weighted move values equal the search value
  (correlation 1.000); the targets turn together with the position; memory save / load / resize keep them;
  a network without the outputs learns as before; a run that is stopped and continued works.

**The test:** `colab_long_game.ipynb`, one Colab session per network, 100 rounds from scratch with large's
settings, compared with large's log (14% at round 20, 28% at 40, 35% at 60, 43% at 80, 48% at 100):

| `NETWORK` | What it shows |
|---|---|
| `large_long` | all three lessons |
| `large_long_no_owner` | without the owner: is the owner lesson worth it? |
| `large_repeat` | large again, nothing changed: how much two equal trainings differ by chance (never measured; large's log is a single run) |

Estimate: 2.5 to 3 hours and 5 to 6 units per network (guessed from large's 3.9 hours and the faster copy).
Command inside the notebook: `python -u src/train.py run <NETWORK> 100`.

**Measured (16 November, L4, 100 rounds each; logs `models/alphazero_large_long_log.csv` and
`models/alphazero_large_repeat_log.csv`).** Points against medium, 256 games per check:

| Round | large (older run) | `large_repeat` | `large_long` | line network |
|---|---|---|---|---|
| 20 | 13.7% | 12.7% | 14.8% | 18.8% |
| 40 | 28.5% | 29.5% | 31.4% | 40.0% |
| 60 | 35.4% | 34.8% | 35.0% | 42.0% |
| 80 | 42.8% | 36.7% | 38.9% | 47.3% |
| 100 | 48.4% | 40.8% | 43.8% | 46.7% |
| mean of all 25 checks | 30.6% | 27.2% | 30.4% | 34.8% |
| mean of the last 10 checks (rounds 64 to 100) | 44.2% | 39.4% | 42.4% | 45.4% |
| hours for 100 rounds | 3.93 | 2.53 | 2.60 | 7.92 |

- **Chance between two equal trainings is about 3 points.** `large_repeat` is the same network and settings
  as large and ends 3.4 points lower on average (below large in 19 of 25 checks, 4.8 points lower over the
  last 10). Not fully equal: large's run is older and played without the faster network copy. Whether that
  matters is not known.
- **The lessons show no gain that stands out from that.** `large_long` is 3.2 points above `large_repeat`
  on average (ahead in 20 of 25 checks) but 0.2 below the older large run. One run each, so a real gain of
  0 to 3 points fits the numbers; a clear gain like the line network's early lead (up to 11 points) is not
  there.
- What the extra outputs learned (loss at round 1 -> 100): owner 1.025 -> 0.911 (pure guessing is 1.099, so
  the final owner of a board is hardly predictable, which fits the user's doubt); result as three chances
  1.028 -> 0.764; move values 0.012 -> 0.022 (flat from round 10).
- Value loss at round 100: `large_long` 0.110, `large_repeat` 0.110, large 0.121. Policy loss 1.333, 1.338,
  1.293. The lessons do not change either.
- Time: the lessons cost about 3% (2.60 against 2.53 hours). Both runs took about 5 units. The faster copy
  makes large's 100 rounds 1.55 times as fast as the older run (2.53 against 3.93 hours).
**Third run, `large_long_no_owner` (16 November, L4, 100 rounds, 2.61 hours, about 5 units; log
`models/alphazero_large_long_no_owner_log.csv`):**

| | large (older run) | `large_repeat` | `large_long` | `large_long_no_owner` |
|---|---|---|---|---|
| Round 20 | 13.7% | 12.7% | 14.8% | 14.6% |
| Round 40 | 28.5% | 29.5% | 31.4% | 30.9% |
| Round 60 | 35.4% | 34.8% | 35.0% | 40.8% |
| Round 80 | 42.8% | 36.7% | 38.9% | 41.2% |
| Round 100 | 48.4% | 40.8% | 43.8% | 42.6% |
| mean of all 25 checks | 30.6% | 27.2% | 30.4% | 30.0% |
| mean of the last 10 checks | 44.2% | 39.4% | 42.4% | 42.0% |

- **The owner lesson neither helps nor hurts.** Without it the run is 0.3 points below `large_long` on
  average (ahead in 10 of 25 checks), far inside the 3 points of chance. It can be dropped.
- **Both lesson runs sit about 3 points above `large_repeat`** (+3.2 and +2.8 on average, ahead in 20 and 21
  of 25 checks) and level with the older large run (-0.2 and -0.6). So the lessons are worth 0 to 3 points
  against medium at 100 rounds; which end of that is true depends on whether `large_repeat` or the older
  large run is the typical one, and that needs more equal runs to tell.
- Without the owner, the result as three chances is learned slightly slower (0.774 against 0.764 at round
  100), the move values the same (0.022). Value loss 0.114, policy loss 1.326 at round 100.
- **Decision:** the lessons are not a step forward that is worth building on by itself. Not taken into the
  main training for now.

## What comes after

1. Open decision: is twice the learning speed per game worth it? Only if the line network runs about as
   fast as large. The faster network copy now covers the line network (16 November: `folded` and
   `_fast_copy` in `src/Tic_Tac_Toe_NN.py`). Checked on the laptop: same answers (move probabilities differ
   by 0.00002), it follows a change of the weights, and the translation into faster code works on the
   processor; 6,150 against 4,450 positions per second on the laptop's chip.
   **Measured on the L4** (`pretest.py bottleneck`, 512 games at once, 400 simulations, ms per move of a game):

   | | as before | rearranged copy | copy, translated |
   |---|---|---|---|
   | line network | 12.6 | 10.4 | 6.8 |
   | large | 7.2 | 5.6 | 4.3 |

   The copy makes the line network's self-play 1.85 times as fast, but large gains 1.67 times too, so the
   line network stays 1.58 times slower per move (was 1.75). The network is 88% of its time. So per hour the
   line layers win little or nothing unless the layers themselves get faster (untried: sum along the lines
   without the big 81 x 72 table multiplications, which fit the card's fast memory layout badly).
   Part 1 of that test ("network alone") shows the line network 5 times slower than large (20,900 against
   107,300 positions per second); that does not fit part 3 and is probably caused by the test's random,
   non-0/1 inputs (guessed: answers that are not a number get asked again in full precision). Not looked into.
   **Tried against that (16 November, speed on the L4 not measured yet):** the copy can build the summaries
   with picture operations instead of the tables (`picture_summary` in `src/test_designs.py`; rows and columns
   by pooling, diagonals with fixed masks). Same answers (differences of 0.000003, also for D8, C2, D5w). On the
   laptop's chip it is slower (4,000 against 6,200 positions per second), so it is switched off
   (`FAST_LINES = False` in `src/Tic_Tac_Toe_NN.py`) until the L4 says otherwise:
   `!python -u src/pretest.py bottleneck large_lines` now shows two more lines in part 3 ("summaries as
   picture operations"). If they beat 6.8 ms, set `FAST_LINES = True`.
   **Measured on the L4:** picture operations, translated: 6.2 ms per move against 6.7 ms with the tables
   (8% faster, 23 s of translating once); without translation they are slower (13.2 against 10.5 ms). Too
   little to close the gap to large's 4.3 ms; `FAST_LINES` stays off for now. Noticed: with them the card is
   busy only 50% of the time while the network call is 89% of the time, so half of that time is not spent
   computing on the card. Suspected (not measured): answers that are not a number in half precision are
   asked again in full precision more often for the line network; that would also explain part 1.
   The speed test now counts this ("asked again in full precision: ... of ... positions").
   **Counted on the L4: the suspicion was wrong for real play.** In self-play 0 of 819,200 positions were
   asked again, in every way of asking. Only part 1 with its random inputs is hit (84% of the positions
   asked again), so part 1 says nothing about the line network; ignore it. Second measurement of the picture
   operations, translated: 5.9 ms against 6.6 ms with the tables (first: 6.2 against 6.7), so about 9% faster
   both times. **Result: the line network is 1.4 to 1.5 times slower than large in self-play (5.9 to 6.6 ms
   against 4.3 ms per move), and that is the cost of the layers, not a fault.** No further speed work planned.
   The network class still lives in `src/test_designs.py`.
2. A direct match `large_lines` against large's round-100 snapshot (both files from Drive) would show
   whether the equal score against medium hides a difference.
3. Continuing the line network past round 100 with 400 simulations (like large) would show whether it
   ends higher than large's 52%. About 9 hours more at today's speed.
4. Lessons about the long game: all three networks run, no clear gain, the owner lesson changes nothing (see the section above).
5. Still open: 1024 games per round from the start; large against minimax on the laptop; D7 at equal
   thinking time; which layer overflows in half precision (optional).

## Settings of the large run (`src/train.py`)

| Round | Time from start | What happens |
|---|---|---|
| 1 to 100 | 0 to 3.9 hours | 200 simulations per move, learning rate 0.001, memory 300,000 positions |
| 100 | 3.9 hours | Simulations go to 400; rounds take about twice as long |
| 120 | 5.5 hours | Learning rate 0.0003 |
| 190 | about 10 hours | Learning rate 0.0001, memory 600,000 |
| 240 | about 14 hours | Saves and stops by itself |

Also: 512 games per round, all at once; every position used 4 times; snapshot every 4 rounds; every 4 rounds
256 games against medium (100 simulations, always the same openings); no minimax during training.

## Decisions and why

- **Pure network.** No minimax in play or training; minimax is only a measuring opponent (the user's preference).
- **From scratch.** A large network taught only from medium's last games scored 19% against medium.
- **Judge by games, not by the loss.** Medium's loss fell from round 61 to 130 while its strength stood still.
  In the design tests the design with the lowest loss (C2) was not the strongest.
- **Learning-rate rule.** Keep the rate while the points against the reference rise from check to check; lower
  it when they are flat. That moved the steps from rounds 80 and 160 to 120 and 190.
- **512 games at once.** Fastest on the L4 (see speed).
- **Search setting (c_puct) stays 1.5** in training and play. A narrower search helps only at short thinking time.
- **Change one thing at a time,** so its effect can be read from the checks.
- **Compare losses only within the same number of simulations;** more simulations lower the policy loss by themselves.

## Results

### Large against medium

Checks during training (256 games, 100 simulations):

| Rounds | 20 | 40 | 60 | 80 | 100 | 80 to 120 | 124 to 160 | 164 to 188 | 192 to 216 | 220 to 240 |
|---|---|---|---|---|---|---|---|---|---|---|
| Points | 13.7% | 28.5% | 35.4% | 42.8% | 48.4% | 46.2% | 51.1% | 52.3% | 53.2% | 54.8% |

(The averages are uncertain by about 1.2 points.) The step of the learning rate at round 120 lifted the checks
by about 5 points; after that they crept up by about 1 point per 25 rounds.

Snapshot test (large snapshot against medium round 240, 256 games, about 2 points uncertain):

| Large round | 40 | 80 | 120 | 160 | 200 | 240 |
|---|---|---|---|---|---|---|
| 500 simulations | 25.8% | 40.6% | 37.7% | 48.4% | 52.1% | 52.3% |
| 2000 simulations | 28.9% | 41.4% | 39.1% | 48.0% | 52.0% | 50.0% |

Reading: the last learning-rate step and the last 40 rounds added nothing measurable. With more thinking time
large looks a little weaker than in the check at 100 simulations, which overstates it by 2 to 4 points. Ties
grow with the rounds and the thinking time (192 of 256 at 2000 simulations). Large scored about 57% moving
first and 43 to 48% moving second.

### Training tests on Colab (large continued from round 240, checks with 512 games)

Baseline, large's last 4 checks played again: 52.7% (+-0.7%). Rule set beforehand: 3 points or more is a real
gain, 1 point or less is none.

| Test | Change | All checks | Last 3 checks |
|---|---|---|---|
| `more_games` (to round 260) | 1024 games per round, each position used 2 times | 55.1% (+-0.7%) | 54.6% (+-0.9%) |
| `more_simulations` (to round 272) | 800 simulations | 52.8% (+-0.5%) | 51.4% (+-0.9%) |

Reading: more games help a little (+2.4 points, likely real, under the 3 points asked for), more simulations
do not. Both ran at the lowest learning rate, so "no gain" is weaker evidence than a gain.

### Medium, the search, and uttt.ai

- **Medium** levelled off from about round 140; round 240 scored about 55% against round 132. Against minimax
  (round 132, 2000 simulations): 96/88/88/76% against depth 12/13/14/15; 4000 simulations added nothing.
- **Search setting** (medium against itself, other setting against 1.5): 0.5 scored 55.1% at 500 simulations
  and 49.4% at 2000; 0.75 scored 54.7% and 50.0%; 2.5 scored 42.4% at 500.
- **Search depth:** the main line grows by 2 to 3 moves per doubling of the simulations (about 12 moves at
  500, 17 at 2000, 20 at 4000).
- **Medium against uttt.ai's network** (236 games, equal simulations; files local only in `external/` and
  `src/test_vs_utttai.py`; results in `models/alphazero_medium_vs_utttai.csv`): with this project's search for
  both, medium scores 47.0% at 500 simulations and 50.6% at 2000, so the networks are level. With their own
  search settings their network plays worse (medium 51.5% and 57.8%). Their network is about 2.2 times slower
  per position than medium.

## Design tests (`src/test_designs.py`)

**Method.** Every design is a new network that learns from large's saved positions (540,000; the newest 60,000
are held back), 6,000 steps of 256 positions at medium size (64 channels, 6 blocks). Measured on the held-back
positions: distance to the search's move probabilities and value error (smaller is better). Then a match of
236 games (118 openings from both sides) at 200 simulations, uncertain by about 2.5 points. Two runs of
today's design differ by about 5 points in the match and 0.009 in distance, so smaller differences mean nothing.
**Limit:** this shows how well a design learns given lessons, not how it trains itself.

| Design | What is changed | Distance | Value error | Against D0 (two runs) | Against D5 (two runs) |
|---|---|---|---|---|---|
| D0 | today's design | 0.285, 0.276 | 0.136, 0.133 | 46.0% | 28.6% |
| D1 | filters stay inside the small board | 0.246 | 0.130 | 59.7%, 56.6% | |
| D2 | small-board summary by a learned step instead of the average | 0.273 | 0.132 | 48.1% | |
| D3 | permanent board track, value read from it | 0.283 | 0.134 | 53.8% | |
| D4 | each board hears from the 9 cells that send into it | 0.267 | 0.133 | 55.5% | |
| **D5** | **line layers** | **0.232, 0.235** | **0.128, 0.121** | **70.6%, 71.0%** | |
| D6 | exact facts as 8 extra input pictures | 0.268 | 0.134 | 61.2%, 56.4% | |
| D7 | whole body on the 3x3 big board (as uttt.ai), 2.6 times as fast | 0.317 | 0.146 | 41.3% | 22.0% |
| C1 | D1 + D5 | 0.241, 0.235 | 0.122, 0.122 | 69.1%, 65.9% | 44.5%, 43.0% |
| C2 | D1 + D4 + D5 + D6 | 0.229, 0.230 | 0.120, 0.119 | 66.7%, 66.1% | 50.8%, 45.1% |

**Reading.** D5 alone is the winner: its gain repeats, and no combination beats it in a direct match although
C2 has the lowest losses. D1 and D6 help a little alone but not on top of D5; D2, D3, D4 are within the noise.

**How the line layers work.** In every block, each cell's description is shortened to 32 numbers, added up
along each of the 72 small lines (fixed table), worked on by a learned step (same weights for all lines), and
each cell gets back the sum of the lines through it. The same is done for the 8 lines of the big board with
the 9 small-board descriptions. Nothing of today's design is removed; 7% more weights.

**Where the errors are** (held-back positions; D0 and D5 are averages of two runs; "large" has learned from
these positions, so its numbers are flattering). Value error, and how often the search's best move is found:

| Kind of position | Share | D0 | D5 | Large |
|---|---|---|---|---|
| no board decided | 44% | 0.090 / 74% | 0.089 / 77% | 0.087 / 84% |
| 1-2 boards decided | 24% | 0.127 / 60% | 0.126 / 62% | 0.107 / 71% |
| 3-4 boards decided | 15% | 0.221 / 56% | 0.213 / 60% | 0.154 / 67% |
| 5+ boards decided | 17% | 0.185 / 54% | 0.140 / 58% | 0.078 / 67% |
| sent to one board | 89% | 0.133 / 66% | 0.127 / 69% | 0.103 / 76% |
| free choice of board | 11% | 0.147 / 53% | 0.112 / 58% | 0.075 / 67% |

Reading: the line layers gain most where the whole board matters (free choice and late game: move distance
about -30%, value error -24%) and nothing for the value in the opening, where all networks have the same error
(noise in the lessons). Hardest for every network: the middle game with 3-4 boards decided, and late free
choices (large finds the best move in 61% of them). Missing in D0 and D5: a summary of the whole board for
every cell (tested tonight as D8) and lessons about the long game.

**Commands:** `speed`, `train [names]` (`D0:1` = second run), `match [names] [vs=D5]`, `summary`, `night`; behind
any of them `size=large` and `steps=36000`. Results: `models/design_tests.csv`, `models/design_matches.csv`;
networks in `models/pretest/` (not in git).

## Speed and cost

- **Colab L4:** 1.94 units per hour, 12 processor cores. Large: 105 to 125 seconds per round at 200
  simulations, 250 to 280 at 400 (before the faster copy). T4: about 4 and 7 minutes. Mac: 16 and 31 minutes.
- **Units are not a tight limit** (the user buys more if needed); state the cost, but do not cut useful tests.
  Units are charged while a session is connected; bought units expire.
- **What limits self-play on the L4:** the network on the card takes 89% of the time, the search tree 11%.
  512 games at once is fastest (13.0 / 7.0 / 7.8 / 10.7 ms per move of a game at 256 / 512 / 1024 / 2048);
  bigger groups are slower per position, cause unknown. GPU and system memory are barely used.
- **Faster network copy** (`FAST_PLAY`, `COMPILE` in `src/Tic_Tac_Toe_NN.py`): on Nvidia cards the search asks
  a rearranged copy (normalisation worked into the filters, other memory layout, half precision, translated
  with `torch.compile`). It is renewed when the weights change, checked against the real network after every
  change, and switches itself off if it disagrees. Measured: network alone 1.9 times as fast; self-play 7.1 to
  4.3 ms per move (1.65 times); real rounds 319 seconds instead of 544 (800 simulations) and 291 instead of
  536 (1024 games). Only today's design is covered. The line "Not enough SMs to use max_autotune_gemm mode"
  is harmless.

## Known problems

- **Half precision fails in rare positions** (a number inside the network gets too big). This ruined the
  network once in round 33. Now those positions are asked again in full precision, damaged positions never
  reach the memory, and learning stops without saving if the loss is not a number.
- **The stop button on Colab can send its signal twice.** Saving ignores further signals; wait for "Saved to ...".
- **A dropped session** loses at most the running round; the memory can be up to 4 rounds older.

**Signs that something is wrong:** a round of large takes more than 150 seconds at 200 simulations (320 at
400, before the faster copy); a message about "not a number"; the points against medium fall over several checks.

## How to

- **Train on Colab:** `colab_train.ipynb`, L4, run the cells from the top. After a disconnect run all cells again.
- **Train another network with the same settings:** `python -u src/train.py run <network> <rounds>`.
- **Use new code:** stop the training cell, run the code cell, run the training cell.
- **Go back to a snapshot:** `python -u src/train.py restore <round>`.
- **Training test:** `colab_experiment_<name>.ipynb`, or `python3 src/train.py experiment <name>` (settings: `EXPERIMENTS`).
- **Snapshots against medium:** `colab_test.ipynb` (multiples of 40, 500 and 2000 simulations, only what is missing).
- **What limits the speed / faster ways on the card:** `python -u src/pretest.py bottleneck` and `... card` (3 minutes each).
- **Chart of a log on the laptop:** `/usr/bin/python3 src/plot_log.py large`.
- **Files on Drive** (`Super-TikTakToe-models`): the networks, `snapshots/`, logs, memories, `experiments/<name>/`.
- **When a run ends:** Runtime > Disconnect and delete runtime, so the units stop.
- **On the laptop** only `/usr/bin/python3` has the packages; long runs with `caffeinate -i`.

## If a bigger network is trained later (`xlarge`: 128 channels, 10 blocks)

Only worth it once something makes size pay off. Then: measure the cost first (`pretest.py speed xlarge`,
about 1.75 times large on the T4); expect the half-precision overflow earlier; do not copy round numbers but
use the learning-rate rule; check against large instead of medium; stop if it is not ahead of large by round 40 to 60.

## What others have built (web research, 15 November 2026)

- **uttt.ai** (github.com/ar-nowaczynski/utttai): about 5 million weights, 256 channels, 3 blocks; its body works
  on the 3x3 grid of small boards. Trained in two stages of about 8 million positions each, with many
  simulations per position. It also learns a value for every move.
- **CodinGame arena** (100 ms per move, no graphics card): tiny networks of about 215,000 weights that only
  judge positions, fast search, over 300 million positions. Their rules break ties differently.
- **Solved?** Only a rule variant is proven a first-player win; our rules are unsolved.
- **Compared with us:** large saw about 6.4 million positions. Others used similar or far more data and a
  design built around the small boards.

## Untested ideas

- A learned description for each of the 19,683 fillings of a small board.
- An attention network with the game's links built in.
- Games that carry over from one round to the next, so more can run at once without bigger rounds.
- A search setting between 1.0 and 1.5 in training; weight decay and the search's value for untried moves were never tuned.
