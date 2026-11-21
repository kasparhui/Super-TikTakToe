# Training notes

Last updated: 21 November 2026 (test results, the final run finished at round 1296 and its end matches; shortened on 17 November; the long version with every check and interim reading is
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
- **Tests for the final model (18 November):** `sims_200` (200 simulations, about 1.7 times the games per hour)
  won its match against `more_games` 260 with 56.8% and passed the rule; `fast_moves` (KataGo's trick) 51.8%, did
  not. The final run (`python3 src/train.py final`, notebook `colab_final.ipynb`) continues `sims_200`.
- **The final run is finished (20 November, round 1296, `models/final/alphazero_large.pt`):** about 1,041,000
  new games after round 279 (37.3 hours of self-play and learning, all Colab units used). In the training checks
  against large 240 (100 simulations, 512 games) it scores about 62% at the end (from about 60% before the late
  learning rate); openings worse about 35 of 256. End matches (laptop, 21 November, 300 different openings from
  both sides): against large 240 61.3% at 500 simulations and 58.2% at 2000, worse in only 22 and 19 of 300
  openings (large 240 against medium 240: 57 and 35); against minimax depth 15 with 2000 simulations 88.2%,
  worse in 1 of 102 openings (see "End matches of the final network").

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

## Proposal for the final model (17 November; tests run 18 November, final run built, see the end of this section)

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
2. Final run, about 75 units (about 39 hours), **in a folder of its own** (the user's wish, 17 November:
   `Super-TikTakToe-models/final/` on Drive, `models/final/` on the laptop): the network, its memory, `logs/`,
   `snapshots/`, `archive/` and its match results all go there; it starts from a copy of the chosen network and
   its memory and writes nothing outside that folder. Large continued from the best of these networks; 1024 games
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

**Speed of large, measured on the L4 (17 November, `fast_moves` session, 400 simulations, faster copy):** 512 games
at once 4.8 ms per move on 1 core, 3.9 ms on 12 cores; 1024 games at once 4.4 and 3.9 ms. So all cores make large
1.1 to 1.2 times as fast, 512 and 1024 at once are equal, and the network is 92 to 94% of the time (the card is the
limit). For the final run: 1024 games at once (no slow tail at the end of a round). `more_games`' round today:
about 1024 x 52 x 3.9 ms = 210 s of self-play + about 50 s of learning = about 260 s (guessed), so the tests'
5,600 s are about 21.5 of its rounds instead of 20: the tests get about 7% more time, a small advantage for them.

**First rounds of the tests (L4, round 241, reported by the user):** `fast_moves` 1024 games, 14,565 positions
learned (27% of the moves), 121 s of self-play; `sims_200` 1024 games, 53,416 positions, 140 s of self-play. So
about 2.1 times (`fast_moves`) and 1.7 times (`sims_200`) the games per hour of `more_games` (guessed 260 s per
round today), about as guessed. Expected rounds in the 1.56 hours: about 45 and 36.

**Game archive for the final run (built 17 November, the user's wish to keep the games for later):**
`ARCHIVE = True` in `src/train.py` writes every iteration's games to `archive/alphazero_<network>_gamesNNNN.npz`
(in the run's folder): moves and result of every game, and for every learned position the visits of the moves
the search tried and its value. `archive_positions(path)` plays the moves again and gives back exactly the
lessons of the memory (board pictures, policy, value). Not kept: the long-game lessons (not used). Checked on the
laptop: rebuilt lessons identical to the last bit, with and without the trick; 13 to 23 bytes per learned
position at 40 simulations against 1,067 in the memory (at 400 simulations more moves are tried, guessed 20 to
30 bytes, so about 0.6 to 1 GB for the final run). Off by default; the final run switches it on.

**Results of the tests (18 November, L4, 5,600 s of self-play and learning each, reported by the user):**

| | rounds | checks against medium (512 games): all / last 3 | match against `more_games` 260 (256 games, 500 sims) | openings worse / equal / better |
|---|---|---|---|---|
| `fast_moves` | 241 to 289 | 53.8% / 54.1% (±0.8) | 51.8% (39-187-30) | 15 / 82 / 21 |
| `sims_200` | 241 to 279 | 55.0% / 56.9% (±0.8) | **56.8%** (54-183-19) | 12 / 68 / 38 |
| `more_games` (for comparison) | 241 to 260 | 55.1% / 54.6% | | |

Baseline (large's last 4 checks up to 240, played again with 512 games): 52.5% (±0.7). Measured: `sims_200` passed
the rule set beforehand (53%) clearly and is better on more than three times as many openings as it is worse;
`fast_moves` did not pass (its 51.8% is inside the chance). Reading (guessed): at this stage more games help more
than a deeper search per move, and KataGo's trick, which learns only from a quarter of the moves, gave too few
lessons per hour to make up for it.

**The final run (built 18 November):** `python3 src/train.py final` (settings in `FINAL` in `src/train.py`; notebook
`colab_final.ipynb`). Large continued from `sims_200` round 279 with its settings (1024 games per round, all at once,
200 simulations, each position learned from twice); memory 1.5 million positions (about 28 rounds); learning rate
0.0001 and, new, `LATE_LEARNING_RATE`: 0.00003 once 85% of the time is used (placed by time, since the number of
rounds is not known beforehand); stops by `TIME_LIMIT` after 36 hours of self-play and learning (about 38 hours
with the checks and restarts, about 75 units at 1.94 per hour). Guessed about 145 s per round, so about 890 rounds
and 900,000 games (7 times large's run). Every 8 rounds a check against large's snapshot 240 (new:
`REFERENCE_ITERATION`; log columns `vs_large240` ...) with 512 games, now also counting the openings worse / equal
/ better (log columns `worse_`, `equal_`, `better_`; every check now counts them), and a snapshot. Archive on. Everything
in `final/` (the first start copies the `sims_200` network, memory and log and large's snapshot 240 there). Drive
space needed, guessed: snapshots about 2.4 GB, memory about 1.6 GB, archive about 1 GB. Tried on the laptop with
a tiny setup: the setup copied only into `final/`, a restart continued and counted the time already used, the late
learning rate began at its share of the time, the archive and the per-opening counts were written.

**Final run, reading at round 436 (18 November morning, log in `models/final/logs/`):** 157 rounds, 5.72 of 36 hours
used (131 s of self-play and learning per round, 165 to 175 s with a check), about 161,000 new games. At this speed
the run ends near round 1270 with about 1 million new games. Checks against large 240 (512 games, 100 simulations,
±1.4 per check), averaged:

| rounds | points | openings worse / equal / better |
|---|---|---|
| 280 to 312 (5 checks) | 51.3% | 65 / 115 / 76 |
| 320 to 376 (8 checks) | 53.7% | 58 / 108 / 90 |
| 384 to 432 (7 checks) | 54.2% | 54 / 113 / 89 |

Measured: about +3 points and about 11 fewer worse openings since the start, but flat since about round 345. Losses flat
(policy 1.22, value 0.104), ties in self-play slightly up (about 530 to about 550 of 1024). Guessed: a plateau at
this learning rate; earlier runs gained most at the learning-rate steps. Decision noted: look again around
round 550; if still flat, consider placing the step to 0.00003 earlier.

**Reading at round 627 (18 November evening):** 12.57 of 36 hours used (129 s per round), about 356,000 new games;
the run ends near round 1280, the step to 0.00003 comes near round 1130. The plateau was not one; the checks rose on:

| rounds | points | openings worse / equal / better |
|---|---|---|
| 440 to 496 (8 checks) | 55.6% | 47 / 116 / 93 |
| 504 to 560 (8 checks) | 56.2% | 51 / 103 / 103 |
| 568 to 624 (8 checks) | 57.3% | 51 / 97 / 109 |

Measured: about +1 point per 60 rounds since round 380 (±0.5 per block of 8 checks), the best check so far 59.8% at
616. Policy loss slightly down (1.223 to 1.210), value loss 0.104 to 0.103, ties in self-play steady at about 549.
But the worse openings stopped falling at about 50 of 256 since round 440: the gain comes from equal openings turning
into better ones, not from fewer mistakes against large 240 (at 100 simulations; the 500 and 2000 simulation
matches at the end will tell more). Decision: nothing changed; the step to 0.00003 stays at 85% of the time.
Units: about 60 left at round 627 (matches the log: 13.0 hours at 1.94 per hour); the rest of the run about 48, the end
matches about 3. Agreed with the user: if the checks still rise, add 3 hours (`TIME_LIMIT` 36 to 39 hours), decided
before hour 30.6 of the limit (otherwise the learning rate would go back to 0.0001 for a while; see `LATE_LEARNING_RATE`).

**Reading at round 862 (19 November):** 21.04 of 36 hours used (130 s per round), about 597,000 new games.

| rounds | points | openings worse / equal / better |
|---|---|---|
| 632 to 695 (8 checks) | 57.0% | 50 / 98 / 108 |
| 696 to 759 (8 checks) | 57.4% | 46 / 101 / 109 |
| 760 to 823 (8 checks) | 58.7% | 45 / 93 / 118 |
| 824 to 856 (5 checks) | 58.1% | 43 / 103 / 110 |

Measured: the points rise more slowly now (about +1 in 200 rounds, before +1 in 60), but the worse openings fall again,
from about 50 to about 43, which is what the user's rule asks for. Policy loss 1.210 to 1.200, value loss 0.103 to
0.100, ties in self-play 550 to 564: all signs of fewer mistakes in its own games. Best checks 60.3% (760) and 60.1%
(784, 792). Guessed: 3 more hours at 0.0001 would add about half a point; the step to 0.00003 matters more.
Decision (the user, 19 November): first add the 3 hours, then, with 42 units left at 21 hours (about 1.98 units per
hour measured; the 39 hours would have left 1 to 2 units), **keep 36 hours** and use the units for tests on Colab.
The step to 0.00003 stays at 30.6 hours, the end near round 1280. Kept from the change: once the late learning rate has
begun (read from the log), it stays even if the limit is raised later, so the time of the restart does not matter.
Tried on the laptop with the tiny final run: limit raised after the step, the rate stayed at 0.00003.

**Reading at round 1021 (19 November):** 27.03 of 36 hours used, about 760,000 new games; rounds a little slower now
(129 s to about 138 s). The step to 0.00003 comes near round 1115, the end near round 1250 (guessed).

| rounds | points | openings worse / equal / better |
|---|---|---|
| 760 to 823 (8 checks) | 58.7% | 45 / 93 / 118 |
| 824 to 887 (8 checks) | 58.7% | 41 / 105 / 110 |
| 888 to 951 (8 checks) | 60.4% | 33 / 104 / 119 |
| 952 to 1015 (8 checks) | 60.3% | 34 / 102 / 120 |

Measured: the points level off at about 60%, but the worse openings fell clearly, from about 45 to about 33 of 256
(best 20 at round 904): by the user's rule the network still improves, it makes fewer mistakes against large 240.
Ties in self-play 554 to 572 of 1024, value loss about 0.099, policy loss 1.199 to 1.191.

**End moved to round 1296 (20 November, the user's decision, at about round 1262):** instead of stopping after 36
hours (that would have been about round 1267), the run ends at round 1296, a round with a check and a snapshot.
Measured from the log up to round 1222: 34.35 of 36 hours used, 130.5 s per round in the last 20 rounds; the late
learning rate began at round 1120. Setting: `ITERATIONS` 1296 and `TIME_LIMIT` raised to 40 hours (only a safety
stop, it is not reached); the late learning rate stays because it has begun. Guessed: about 30 rounds more than the
36 hours, about 1.1 hours of work, about 1.3 hours with checks, about 2.5 units. The running process still has the
old code: it stops by itself at 36 hours (it saves the memory then); after that a `git pull` and the same start
cell continue it to 1296. Tried on the laptop with the tiny final run: stopped by time, then continued with a raised
limit and a set end round; the rate stayed at 0.00003 and it stopped exactly at that round.

**Lost rounds 1169 to 1260 (20 November):** Google Drive was full. The Colab machine stopped without a proper stop at
round 1262 (35.8 hours); Drive had only the network of round 1168, because files are written to the machine and
uploaded in the background, and with a full Drive the uploads fail without a message. The restart continued from
1168; meanwhile the user freed space, after which the network and log of 1260 seemed to arrive, but the running
restart overwrote the network with its rounds 1169 to 1173, and the 1260 file was not in Drive's trash. So the run
continues from 1173; about 87 rounds (about 3.2 hours, about 6.5 units) are lost; snapshots 1176 to 1256 are missing.
The log has the lost rows (1169 to 1260) and then the new ones; they are to be removed from it (they belong to a
network that no longer exists). The end stays at 1296 (the user's decision, 10 units left, about 9.5 needed,
guessed). The safety time limit is raised to 48 hours, because the log also counts the lost rounds (about 36 hours
counted at the restart, plus about 4.5 hours to 1296 would have passed 40 hours near round 1285). Lesson: keep at
least 3 GB free on Drive (the memory file of 1.6 GB is briefly there twice while it is saved).
Measured in the log (the user's copy at round 1179): the network on Drive was really of round 1167, so the restart
began with round 1168 (the restart's start line said 1176 only after its short first part). The log had the old
rows 1168 to 1222 (Drive's log had arrived only up to 1222) and then the new rows from 1168. Cleaned: the old rows
are moved to `models/final/logs/alphazero_large_log_lost_1168_to_1222.csv`, the log is continuous again (1 to 1179).
Drive's log is left as it is while the run goes on; the same cleaning is done again at the end. Checks against
large 240 of the lost rounds 1168 to 1216 (7 checks): 62.6% on average, worse openings about 29; of the late
rate's rounds kept, 1120 to 1176 (8 checks): 61.2%, worse openings about 32. Guessed: the difference is mostly
chance (one check has 512 games, about +-2 points), but the lost rounds hint that the late rate still adds a little.

**End of the final run (20 November, round 1296):** the user's copies of the network (iteration 1296) and log;
Colab units used up. The log again had the lost rows 1168 to 1222 (the same as in the file of lost rows); removed,
the log is continuous from 1 to 1296. 37.29 hours of self-play and learning after round 279 (38.55 with the checks),
1,017 rounds, 1,041,000 games; 0.00003 in every round from 1120. (The network file's optimizer shows the learning
rate 0.001: guessed, the start cell was run once more after the end, which loads the file, sets the starting rate,
plays no round and saves again; the weights are the same, and the rate is set again in every round anyway.)

| rounds | checks | points against large 240 | openings worse / equal / better (per check) | ties of 512 |
|---|---|---|---|---|
| 888 to 951 | 8 | 60.4% | 33.2 / 104.1 / 118.6 | 272 |
| 952 to 1015 | 8 | 60.3% | 33.6 / 102.0 / 120.4 | 276 |
| 1016 to 1079 | 8 | 59.7% | 36.8 / 98.6 / 120.6 | 278 |
| 1080 to 1119 | 5 | 61.0% | 33.2 / 95.0 / 127.8 | 271 |
| 1120 to 1176 (0.00003) | 8 | 61.2% | 31.6 / 102.2 / 122.1 | 287 |
| 1177 to 1240 | 8 | 62.5% | 31.8 / 90.5 / 133.8 | 276 |
| 1241 to 1296 | 7 | 61.9% | 35.7 / 92.0 / 128.3 | 273 |

Last check (round 1296): 62.1%, 38 / 84 / 134. Measured: the late learning rate lifted the points from about 60%
to about 62% (the lost rounds 1168 to 1216 also had 62.6%), but the worse openings did not fall further (about
32 to 36, best 20 at round 904). Policy loss about 1.19, value loss about 0.098, ties in self-play about 570 of
1024 in the last 20 rounds: flat since about round 1000. Guessed: the network is near the end of what this size
and these settings give; more rounds would add little.

**End matches of the final network (20 to 21 November, laptop, `src/test_final.py`, results in
`models/final/results/`):** about 16 hours (3.4 of them lost because the laptop slept with the lid closed;
`caffeinate` does not stop that, and `-s` works only on mains power).
- Openings without repetitions (the user's wish): the board has 8 symmetries (turning, mirroring); openings that
  are the same position after one of them count once. Measured: the old tournament's 128 openings are only 71
  positions, medium 132's 83 minimax openings only 53. There are 102 different 2-move openings and 822 different
  3-move ones. Between the networks: all 102 of 2 moves plus 198 of 3 moves (fixed random choice) = 300, each from
  both sides, 600 games per pair and setting. Against minimax: all 102 of 2 moves, the first 53 are medium 132's.
  Without randomness the same opening and side would repeat the same game.
- Between the networks (both with the same simulations; openings worse / equal / better by the user's rule):

| pair | 500 simulations | 2000 simulations |
|---|---|---|
| final 1296 against large 240 | 61.3% (170-395-35), 22 / 144 / 134 | 58.2% (122-455-23), 19 / 173 / 108 |
| final 1296 against medium 240 | 64.0% (196-376-28), 17 / 125 / 158 | 64.2% (188-395-17), 9 / 133 / 158 |
| large 240 against medium 240 (yardstick) | 53.4% (117-407-76), 57 / 154 / 89 | 54.0% (98-452-50), 35 / 179 / 86 |

- Against minimax (204 games each; per opening worse / equal / better from the network's side; the medium 132
  columns are on the same 53 openings, 106 games):

| simulations, depth | final 1296, all 102 openings | final 1296, the 53 | medium 132, the 53 |
|---|---|---|---|
| 500, 13 | 88.0% (163-33-8), 3 / 6 / 93 | 90.1%, 0 / 2 / 51 | 73.1%, 3 / 17 / 33 |
| 500, 15 | 79.9% (133-60-11), 5 / 16 / 81 | 77.8%, 3 / 9 / 41 | 62.3%, 10 / 13 / 30 |
| 2000, 13 | 94.1% (182-20-2), 0 / 2 / 100 | 95.8%, 0 / 1 / 52 | 84.9%, 0 / 6 / 47 |
| 2000, 15 | 88.2% (159-42-3), 1 / 7 / 94 | 89.2%, 0 / 4 / 49 | 75.5%, 3 / 13 / 37 |

- Measured: the final network is clearly the strongest; more simulations do not shrink its lead over large 240
  (worse openings 22 at 500, 19 at 2000), and against medium 240 it is worse in only 9 of 300 openings at 2000.
  Against minimax it loses 2 to 11 of 204 games; medium 132 lost about 3 times as often on the same openings.
  Guessed: about +-1.5 points for 600 games, about +-2.5 for 204. medium 132 is smaller and much shorter trained,
  so the minimax comparison shows the progress of the whole project, not of the final run alone.
- Not done: weight averaging of the last snapshots (they are only on Drive); large 240 against minimax on the
  same openings (would give a direct yardstick, about 6 to 7 hours on the laptop).

**Symmetry of the networks (measured 21 November):** 400 positions from random games, each given in all 8
turnings / mirrorings. Spread of the value (scale -1 to 1) mean / 90% / largest: final 1296 0.09 / 0.16 / 0.30,
large 240 0.12 / 0.20 / 0.39, medium 240 0.11 / 0.20 / 0.36; the move distributions differ by 6.7%, 7.8% and 9.1%
on average; the same best move in all 8 in 59% to 63% (often two moves are almost equal). So: about symmetric,
not exactly; the training only turns each batch of positions by one random symmetry (`random_symmetry`, each
position is learned about twice), the network design does not force it. Not tried, guessed to help most for the
least: turning the position randomly at every network question in the search (as AlphaGo Zero and KataGo do),
free in time; could be tested on the final network on the laptop (about 40 minutes).

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
- Measured (log `models/logs/alphazero_small_many_log.csv`): 83 s per round at 200 simulations, 139 s at 400
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
  At 2000 simulations against depth 15: 108 wins, 51 ties, 17 losses of 176 (`models/results/alphazero_medium_vs_minimax.csv`;
  the moves of those games were not saved).
- **Endgame check** (`src/test_endgame.py`, 12 November; `models/results/alphazero_medium_endgame.csv`): medium round
  132 against **itself**, 64 games, 2000 simulations; minimax depth 14 checked 2075 network moves from move 20.
  Missed proven win: 7 of 109 moves where a win was proven (some may only be slower wins). Moved into a proven
  loss when nothing was proven before: 8 of 1880. Games: 48 ties, 16 decisive. So traps are rare
  but not absent. Not run against minimax (the script can: `OPPONENT_DEPTH = 15`).
- **Search setting** against 1.5: 0.5 scored 55.1% at 500 and 49.4% at 2000; 0.75 54.7% / 50.0%; 2.5 42.4%.
- **Search depth:** the main line grows by 2 to 3 moves per doubling of simulations (12 at 500, 17 at 2000).
- **Against uttt.ai's network** (236 games, our search for both): 47.0% at 500, 50.6% at 2000, so level.
  Their network is 2.2 times slower per position. (Files local only in `external/`; results in
  `models/results/alphazero_medium_vs_utttai.csv`.)

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
- **Direct match** (round 160, 256 games per pair, `models/results/alphazero_large_lines_tournament.csv`):

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
  (`size=large`, `steps=36000`); results `models/results/design_tests.csv`, `models/results/design_matches.csv`.
- **Chart of a log:** `/usr/bin/python3 src/plot_log.py <network>`.
- **The models folder** (tidied 17 November): networks (`alphazero_<network>.pt`) and memories (`_buffer.npz`)
  directly in it; training logs in `logs/`; matches and tests in `results/`; plus `snapshots/`, `experiments/`,
  `archive/`, `pretest/`. `NN.data_path` finds a log or result file also where an older copy of the folder still
  has it directly in the models folder, and keeps using it there, so the flat Drive folder and runs going on
  there keep working; new files go into `logs/` and `results/`. Checked on the laptop: every script finds the
  moved files; a run in a flat folder went on in its flat log; a new run and a test wrote into `logs/`.
- **Files on Drive** (`Super-TikTakToe-models`): networks, `snapshots/`, logs, memories (still flat; no need to
  move them).
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
