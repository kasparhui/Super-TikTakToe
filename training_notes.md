# Training notes: the large network on Google Colab

Last updated: 15 November 2026, after both Colab training tests (more_games, more_simulations) finished and the first design matches on the laptop.
This file is the memory of the project's training decisions. It is kept up to date
whenever something that influences the training changes.

## Where things stand

- The **large** network (96 channels, 8 blocks, 1.84 million weights) is **finished** (round 240, trained from scratch on Colab, L4).
- **Result: large is level with medium.** Snapshot test of round 240 against medium round 240: 52.3% at 500 simulations, 50.0% at 2000 (256 games each, each uncertain by about 2 points). In the check during training (100 simulations) the last six checks average 54.8%.
- **Decision: no bigger network for now.** 2.75 times the weights and twice the simulations in training gave almost nothing, so size is not the limit. Next: the two training tests on Colab (more games, more simulations), see "Tests for how to continue".

## The plan (settings in `src/train.py`)

| Round | Time from start | What happens |
|---|---|---|
| 1 to 100 | 0 to about 3.5 hours | 200 simulations per move, learning rate 0.001, memory 300,000 positions |
| 100 | about 3.5 hours | Simulations go to 400. Rounds take about twice as long |
| 120 | about 5 hours | Learning rate drops to 0.0003 |
| 190 | about 9 to 10 hours | Learning rate drops to 0.0001, memory doubles to 600,000 |
| 240 | about 14 hours | The training saves and stops by itself. Disconnect the runtime afterwards |

Other settings: 512 games per round, all 512 at once; every position used 4 times for learning;
snapshot every 4 rounds; every 4 rounds 256 games against the finished medium network
(100 simulations, always the same openings); no test against minimax during training.

## Decisions and why

- **Large, not bigger.** In a comparison on medium's games, large learned only 3 to 5% more per step than medium. Bigger (`xlarge`: 128 channels, 10 blocks, about 1.75 times the cost on the T4) only after large has clearly beaten medium.
- **From scratch.** A large network taught only from medium's last games scored 19% against medium, so that start saves almost nothing.
- **Pure network.** No minimax in play or training; minimax is only a measuring opponent (your preference).
- **Learning-rate step moved from round 80 to 120** (and 160 to 190). Rule used: as long as the points against medium keep rising from check to check, the current rate still works. From round 60 to 76 they rose from 35% to 47%. From round 76 to 116 they were flat, so the step at 120 stays (decided at round 116).
- **Judge by games, not by the loss.** In medium's run the loss fell from round 61 to 130 while its strength stood still.
- **512 games at once.** On the L4 almost twice as fast as 256. More would need bigger rounds and change the training.
- **Search setting (c_puct) stays 1.5 in training.** A narrower search plays better at 500 simulations but confirms the network instead of testing it; untested for training.

## Results so far

Large against the finished medium network during training (256 games from round 12 on, 100 simulations):

| Round | Points | Wins-ties-losses | Policy loss | Medium's policy loss at that round |
|---|---|---|---|---|
| 20 | 13.7% | 16-38-202 | 1.605 | 1.593 |
| 40 | 28.5% | 27-92-137 | 1.427 | 1.456 |
| 60 | 35.4% | 40-101-115 | 1.354 | 1.388 |
| 76 | 47.1% | 62-117-77 | 1.324 | 1.354 |
| 80 | 42.8% | 50-119-87 | 1.316 | |
| 84 | 45.7% | 61-112-83 | 1.307 | |
| 88 | 45.7% | 63-108-85 | 1.302 | |
| 92 | 44.7% | 59-111-86 | 1.295 | |
| 96 | 47.3% | 61-120-75 | 1.295 | |
| 100 | 48.4% | 58-132-66 | 1.293 | |
| 104 | 49.6% | 67-120-69 | 1.290 | |
| 108 | 42.2% | 46-124-86 | 1.283 | |
| 112 | 46.3% | 54-129-73 | 1.266 | |
| 116 | 46.5% | 49-140-67 | 1.263 | |

| 120 | 49.0% | 56-139-61 | 1.248 | |
| 124 | 52.3% | 71-126-59 | 1.238 | |
| 128 | 49.6% | 65-124-67 | 1.225 | |
| 132 | 50.4% | 60-138-58 | 1.216 | |
| 136 | 48.8% | 63-124-69 | 1.214 | |
| 140 | 48.8% | 57-136-63 | 1.210 | |
| 144 | 53.5% | 65-144-47 | 1.211 | |
| 148 | 51.2% | 64-134-58 | 1.211 | |
| 152 | 52.9% | 71-129-56 | 1.207 | |
| 156 | 52.1% | 68-131-57 | 1.205 | |
| 160 | 50.8% | 65-130-61 | 1.205 | |
| 164 | 51.8% | 67-131-58 | 1.201 | |
| 168 | 51.8% | 61-143-52 | 1.202 | |
| 172 | 50.0% | 64-128-64 | 1.195 | |
| 176 | 50.0% | 60-136-60 | 1.195 | |
| 180 | 54.1% | 74-129-53 | 1.191 | |
| 184 | 53.9% | 73-130-53 | 1.187 | |
| 188 | 54.7% | 75-130-51 | 1.188 | |
| 192 | 52.1% | 71-125-60 | 1.184 | |
| 196 | 53.9% | 72-132-52 | 1.186 | |

Reading at round 196: the checks went from about 51% to about 54% at round 180, ten rounds before the
learning-rate step at 190, and stayed there (average of rounds 180 to 196: 53.7%). The user reports
54% up to the end of the run.

Reading at round 174: the learning-rate step at 120 lifted the checks by about 5 points (average 45.9%
for rounds 80 to 116, 51.1% for rounds 124 to 172, 51.8% for rounds 144 to 172). Since round 144 they
are flat again. 51 to 52% is about 10 Elo above medium: equal within the noise of one check, slightly
ahead on average. The step at round 190 may add a little; in medium's run the last step added about
40 Elo over 108 rounds. Rounds take about 285 seconds now.

Reading at round 116: ten checks in a row (rounds 80 to 116) average 45.9% with no trend, so the
learning rate of 0.001 has stopped giving strength. The loss fell after the step to 400 simulations
(1.293 to 1.263), as expected from sharper targets, without a gain in the check.

Snapshot test on Colab (large snapshot against medium round 240, 256 games each, the same 128 openings
from both sides; each number is uncertain by about 2 points):

| Large round | 500 simulations | Wins-ties-losses | 2000 simulations | Wins-ties-losses |
|---|---|---|---|---|
| 40 | 25.8% | 11-110-135 | 28.9% | 18-112-126 |
| 80 | 40.6% | 29-150-77 | 41.4% | 31-150-75 |
| 120 | 37.7% | 23-147-86 | 39.1% | 22-156-78 |
| 160 | 48.4% | 44-160-52 | 48.0% | 32-182-42 |
| 200 | 52.1% | 51-165-40 | 52.0% | 39-188-29 |
| 240 | 52.3% | 45-178-33 | 50.0% | 32-192-32 |

Reading (15 November): with more thinking time large looks a little weaker against medium than in the
check at 100 simulations (round 160: 48% here, about 52% there), so longer search does not favour the
bigger network. Rounds 80 and 120 are equal within the noise, round 160 shows the gain from the
learning-rate step at 120. At round 160 large is level with medium, not ahead. In all eight matchups
large scored 7 to 15 points more when it moved first than when it moved second.

Final reading (15 November, large finished): rounds 200 and 240 are equal within the noise, so the last
learning-rate step (round 190) and 40 more rounds added nothing measurable here. Large ends level with
medium: about 52% at 500 simulations, 50% at 2000. Ties grow with the rounds and with thinking time
(192 of 256 at 2000 simulations). Large round 240 scored about 57% moving first and 43 to 48% moving second.

Checks during training (100 simulations, 256 games each), averaged: rounds 80 to 120 46.2%, 124 to 160
51.1%, 164 to 188 52.3%, 192 to 216 53.2%, 220 to 240 54.8% (each average uncertain by about 1.2 points).
So the check kept creeping up by about 1 point per 25 rounds, while the snapshot test with more thinking
time stayed flat. The check at 100 simulations overstates large by about 2 to 4 points.

Medium for reference: levelled off from about round 140; round 240 scored about 55% against round 132.
Against minimax (round 132): 2000 simulations gave 96/88/88/76% against depth 12/13/14/15; 4000 added nothing.

Search test with the medium network against itself (256 games each, other setting against 1.5):

| Setting | 500 simulations | 2000 simulations |
|---|---|---|
| 0.5 | 55.1% | 49.4% |
| 0.75 | 54.7% | 50.0% |
| 1.0 | 53.9% | not played |
| 2.5 | 42.4% | not played |

Conclusion: a narrower, deeper search helps only when thinking time is short; at 2000 simulations
it makes no difference. The setting stays at 1.5 everywhere for now.

The main line of the search grows by about 2 to 3 moves per doubling of the simulations
(about 12 moves at 500, 17 at 2000, 20 at 4000 in the middle of the game).

## Speed and cost

- L4: about 105 to 125 seconds per round at 200 simulations, 250 to 280 seconds at 400 simulations (measured from round 101 on); 1.94 compute units per hour (shown by Colab at round 100; earlier assumed 1.7); the whole plan about 25 to 30 units (2.5 to 3 CHF). At round 100, 89.5 units were left; rounds 101 to 240 plus the snapshot tests should cost about 23 to 24.
- T4: about 4 and 7 minutes per round; about 30 units for the plan.
- Mac: 16 and 31 minutes per round for large.
- 100 units were bought; about 65 should be left after the large training and its tests.

- **What limits self-play on the L4 (measured 15 November 2026, `pretest.py bottleneck`, large, 400 simulations, 12 processor cores):**
  the network on the card takes 89% of the time, the search tree on one processor core only 11%. So more
  processor cores would gain at most about 10%. Network alone: 35,700 positions per second at 256 at once,
  62,800 at 512, 56,200 at 1024, 40,900 at 2048. Self-play: 13.0 / 7.0 / 7.8 / 10.7 ms per move of a game at
  256 / 512 / 1024 / 2048 games at once. So 512 is the best size, and bigger groups are slower per position;
  why is not known. The card reports being busy 78% of the time at 512. Next lever: make the network itself
  cheaper on the card (settings of the graphics library, a faster design), not more games at once.

- **Faster ways to run the network on the L4 (measured 15 November 2026, `pretest.py card`, large, positions per second at 512 / 1024 / 2048 at once):**
  as now 66,400 / 58,000 / 41,900; all rearrangements together (normalisation worked into the filters, other
  memory layout, whole network in half precision) 94,400 / 85,300 / 65,200 = 1.4 times; the same translated
  into faster code (`torch.compile`) 124,600 / 113,700 / 87,000 = 1.9 times. Move probabilities differ from full
  precision by 0.002 to 0.003, as now. Copying to and from the card is under 7% of a question; the network
  itself gets slower per position in bigger groups, cause not known.
- **Built into the search (15 November 2026, `FAST_PLAY` and `COMPILE` in `src/Tic_Tac_Toe_NN.py`):** on Nvidia
  cards the search asks a rearranged, translated copy of the network. The copy is brought up to date whenever
  the weights change, is compared with the network itself at the first question after every change, and switches
  itself off with a message if it disagrees. Checked on the laptop for equal answers.
  **Measured on the L4 (15 November 2026, `pretest.py bottleneck`, 512 games, 400 simulations):** 7.1 ms per move
  of a game as before, 5.5 ms with the rearranged copy, 4.3 ms with the translation = 1.65 times as fast, no
  warning printed. The search tree is now 17% of the time (before 11%). 1024 games at once are now as fast as
  512 (4.4 ms), 2048 slower (5.4 ms). In the real training (test `more_simulations`, 512 games, 800 simulations):
  319 seconds per round instead of 544 = 1.7 times as fast (round 251, the first with the faster path).
  Test `more_games` (1024 games, 400 simulations): 291 seconds instead of 536 = 1.84 times as fast (round 249).
  `pretest.py bottleneck` part 3 measures it. Only today's design is covered; a new design needs its own copy rule.

## Known problems

- **Half precision fails in rare positions** (13 of 249,000 at round 32, more later): a number inside the network gets too big and the answer is "not a number". This ruined the network in round 33. Now those positions are asked again in full precision, damaged positions never reach the memory, and learning stops without saving if the loss is not a number. Which layer overflows is not known.
- **Rounds went from 105 to about 120 seconds** after round 37; cause not found; accepted.
- **The stop button on Colab can send its signal twice.** Saving now ignores further signals. Wait for "Saved to ...".
- **A dropped session** loses at most the running round; the memory can be up to 4 rounds older.

## How to

- **Train:** `colab_train.ipynb`, L4, run the cells from the top. After a disconnect run all cells again.
- **Use new code:** stop the training cell, run the code cell, run the training cell.
- **Go back to a snapshot:** `!python -u src/train.py restore <round>`
- **Run a training test:** the notebook `colab_experiment_<name>.ipynb`, or `python3 src/train.py experiment <name>`. After a disconnect run all cells again.
- **Find what limits the speed:** `!python -u src/pretest.py bottleneck` (about 3 minutes; shows processor cores, how the time splits between search tree and network, and 256 to 2048 games at once). Stop the training cell first; it continues afterwards.
- **Try faster ways to run the network on the card:** `!python -u src/pretest.py card` (about 3 minutes; shows where the time of one question goes and the speed of the same network with other settings of the graphics library). Changes nothing in the training.
- **Test snapshots against medium:** `colab_test.ipynb` (multiples of 40, 500 and 2000 simulations, 256 games, only what is missing).
- **Chart of the log on the laptop:** copy `alphazero_large_log.csv` into `models/`, then `/usr/bin/python3 src/plot_log.py large`.
- **Files on Drive** (`Super-TikTakToe-models`): `alphazero_large.pt` (newest), `snapshots/`, the log, the memory, `alphazero_medium.pt` (needed for the checks).
- **When it ends:** after round 240 it prints "Finished: ..." and stops. To train further, raise `ITERATIONS` in `src/train.py`.
- **After stopping:** Runtime > Disconnect and delete runtime, so the units stop.

## Signs that something is wrong

- A round at 200 simulations takes more than 150 seconds, or more than 320 at 400 simulations.
- The training stops with a message about "not a number".
- The points against medium fall over several checks in a row.

## Plan for what comes next

### 1. Finish the large network (done, 15 November 2026)

- Let it run to round 240 even if it passes 50% against medium early. How far above medium it ends decides the next step.
- Done at round 116: the checks were flat, the learning-rate step at round 120 stays.
- Seen at round 174: the step lifted the points to about 51 to 52%, then flat. By the table in step 3 this points to "little gain from size" unless the snapshot test at 500 and 2000 simulations shows more.
- The step from 200 to 400 simulations at round 100 lowers the policy loss by itself (sharper targets), so compare losses only within the same number of simulations.

### 2. Measure the finished large network

- On Colab: the snapshot test (`colab_test.ipynb`) for all multiples of 40, at 500 and 2000 simulations.
- On the laptop: the final network against minimax on the saved openings, as for medium.
- The check during training uses only 100 simulations; the snapshot test is the firmer measure.

### 3. Decide about a bigger network

Decided 15 November 2026: large ended at 50 to 52% against medium, the row "little gain from size". No
bigger network; the training tests below come first. Still open from step 2: large against minimax on the laptop.

| Final result of large against medium (500 and 2000 simulations) | Meaning | What to do |
|---|---|---|
| 65% or more | size helps a lot | train the bigger network |
| about 55 to 65% | size helps somewhat | bigger is reasonable, expect a smaller gain |
| about 50 to 55% | little gain from size | do not go bigger; look at other levers first |
| still rising at round 240 | not finished | train large further (raise `ITERATIONS`) before deciding |

### 4. If the bigger network is trained (`xlarge`: 128 channels, 10 blocks, about 4 million weights)

- **Cost:** about 1.75 times large per round on the T4; measure on the L4 first (`pretest.py speed xlarge`). With about 65 units left, about 240 rounds should fit, but check the estimate before starting.
- **Memory on Colab is barely used** (you, 15 November 2026: GPU memory and system memory are mostly free, and units are not a tight limit). So more games at once and a longer memory of positions are both possible in the next run.
- **Use the card better:** let games carry over from one round to the next, so 1024 or more run at once without making the rounds bigger. This needs a change to the self-play code and a test first. Measure 1024 and 2048 at once with the speed test.
- **Half precision:** expect the overflow problem again, probably earlier. The repair is in place; finding the layer that overflows and running only that layer in full precision would remove the cause.
- **Schedule:** do not copy round numbers. Use the same rule as for large: keep the learning rate while the points against the reference keep rising, lower it when they go flat.
- **Reference network:** the check should then be against the finished large network (set `REFERENCE = 'large'`), with medium as a second line if wanted.
- **Compare with large's log** at the same rounds while the settings are the same; if the bigger network is not ahead of large by about round 40 to 60, stop and save the units.
- **Playing speed on the laptop:** the bigger network needs about twice the time per move of large.

### Tests for how to continue (15 November 2026; step 0 is done, step 1 is built, pushed and ready to run on Colab)

0. **Baseline (done):** snapshot test of large rounds 200 and 240 against medium: 52.1/52.0% and 52.3/50.0%.
   Still to copy from Drive to the laptop: `alphazero_large.pt` and `alphazero_large_buffer.npz` (needed for step 2).
1. **Training tests on Colab,** each continuing large from round 240 in its own Drive folder
   (`Super-TikTakToe-models/experiments/<name>`), checks with 512 games. Settings: `EXPERIMENTS` in `src/train.py`.
   - `more_games` (`colab_experiment_more_games.ipynb`): 1024 games per round, every position used 2 times;
     rounds 241 to 260, about 3 hours, 6 units.
   - `more_simulations` (`colab_experiment_more_simulations.ipynb`): 800 simulations, 512 games, used 4 times;
     rounds 241 to 272, about 5 hours, 9 to 10 units. It runs longer because its memory fills with new games more slowly.
   - Each test first plays large's checks of rounds 228 to 240 again with 512 games (the baseline on the same
     openings) and prints at the end "before the test" against "its last 3 checks".
   - Judge: 3 points or more above the baseline is a real gain, 1 point or less is none (each average is
     uncertain by about 1 point). At the low learning rate "no gain" is weaker evidence than a clear gain.
   - Needs: large ended at round 240 and nothing trained after it (the test refuses to start otherwise).
   - **Results (15 November 2026, both finished).** Baseline, large's 4 checks up to round 240 played again:
     52.7% (+-0.7%, 2048 games).
     - `more_games` (to round 260): all 5 checks 55.1% (+-0.7%, 2560 games), last 3 checks 54.6% (+-0.9%).
       That is +2.4 points over all checks and +1.9 over the last 3: between the two limits. A small gain,
       likely real (about 2.5 times the uncertainty over all checks), but not the 3 points asked for.
     - `more_simulations` (to round 272): all 8 checks 52.8% (+-0.5%, 4096 games), last 3 checks 51.4% (+-0.9%).
       No gain (+0.1 and -1.3 points).
     - Reading: at this stage more games per round help a little, more simulations per move do not. Neither
       lifts large clearly above medium, so training settings alone are not the way forward. For the next
       run: 1024 games per round is the better use of time than 800 simulations (with the faster network copy
       a 1024-game round at 400 simulations takes about 290 seconds on the L4). Both tests ran at the low
       learning rate 0.0001, so "no gain" is weaker evidence than a gain.
2. **Design tests on the laptop** (free; list refined 15 November 2026, not started): same positions (large's
   memory of 600,000, last 10% held back), same steps; measure the error on the held-back positions for move
   and value separately, the positions per second, and a 256-game match against the baseline design.
   Because games are the limit, a design counts as better if it is as good and faster, not only if it is more exact.
   - D0: today's design, twice (shows the noise).
   - D1: filters that stay inside the small board (no looking across borders), same cost.
   - D2: the small-board summary made by a learned 3x3 step that knows which cell is which, instead of the average.
   - D3: a permanent, wide board track (9 small-board descriptions kept through the network), value read from it.
   - D4: the reverse of the sending rule (each board hears from the 9 cells that send into it).
   - D5: line layers (8 lines per small board, 8 lines of the big board, shared weights).
   - D6: exact facts as input (dead stones blanked, cells that win a small board at once, boards that would win
     the game, free-choice flag).
   - D7: a body on the 3x3 big board only, as uttt.ai (the extreme of D2/D3; much faster per position).
   - Then the winners combined, at medium and large size.
   - Second round, needs about 6,000 new games that record more: win/tie/loss as three chances, final owner of
     each small board, the search's value per move, the opponent's reply. (The saved value is a mix of result
     and search value, so win/tie/loss cannot be taken from the old memory.)
   - Already in place, not a test: the 8 turns and mirrors of the board are used in learning.
   - Built 15 November 2026 as `src/test_designs.py` (commands: speed, train, match, summary; results in
     `models/design_tests.csv` and `models/design_matches.csv`). All designs take the usual input, so they
     work in the search unchanged. Screening is at medium size (64 channels, 6 blocks), 6000 steps.
   - Speed measured on the laptop (positions per second, asked 512 at a time): D0 13,300; D1 9,000 (slow
     only because of how it is built); D2 12,700; D3 13,000; D4 13,100; D5 10,400; D6 13,000; D7 35,000.
     Learning takes 6 to 9 minutes per design (D7 3 minutes).
   - Limit of this screening: it shows how well a design learns given lessons, not how it trains itself; step 3 checks that.
   - **First results (15 November 2026, medium size, 6000 steps; distance to the search's move choices and
     value error on the 60,000 held-back positions, smaller is better):**

     | Design | Distance | Value error |
     |---|---|---|
     | D0 today's design (two runs) | 0.2848, 0.2757 | 0.1361, 0.1330 |
     | D1 filters inside the small board | 0.2460 | 0.1297 |
     | D2 learned small-board summary | 0.2732 | 0.1319 |
     | D3 permanent board track | 0.2826 | 0.1340 |
     | D4 reverse of the sending rule | 0.2668 | 0.1331 |
     | D5 line layers | 0.2323 | 0.1284 |
     | D6 exact facts as input | 0.2680 | 0.1335 |
     | D7 body on the 3x3 big board | 0.3173 | 0.1459 |

     Reading: the two baseline runs differ by 0.009, so D1 and D5 are clear gains, D4 and D6 small ones at
     the edge of the noise, D2 and D3 none. D7 is clearly less exact but 2.6 times as fast.
   - **Matches against D0 (first run), 236 games each, 200 simulations, each uncertain by about 2.5 points; the
     second D0 run scored 46.0%, so differences under about 5 points mean nothing:**

     | Design | Distance | Value error | Points against D0 | Wins-ties-losses |
     |---|---|---|---|---|
     | D5 line layers | 0.2323 | 0.1284 | 70.6% | 127-79-30 |
     | C1 = D1 + D5 | 0.2406 | 0.1221 | 69.1% | 119-88-29 |
     | C2 = D1 + D4 + D5 + D6 | 0.2288 | 0.1202 | 66.7% | 105-105-26 |
     | D6 exact facts | 0.2680 | 0.1335 | 61.2% | 101-87-48 |
     | D1 in-board filters | 0.2460 | 0.1297 | 59.7% | 88-106-42 |
     | D4 reverse sending | 0.2668 | 0.1331 | 55.5% | 79-104-53 |
     | D3 board track | 0.2826 | 0.1340 | 53.8% | 85-84-67 |
     | D2 learned summary | 0.2732 | 0.1319 | 48.1% | 70-87-79 |
     | D0 second run | 0.2757 | 0.1330 | 46.0% | 54-109-73 |
     | D7 big-board body | 0.3173 | 0.1459 | 41.3% | 54-87-95 |

     Reading: the line layers are the big gain (about 150 Elo over today's design at this small scale).
     Exact facts and in-board filters help by themselves (about 60%), but adding them to the line layers gives
     no further gain in the match, although the combinations have the best losses. D7 loses at equal
     simulations; at equal time (it is 2.6 times as fast) it is not tested. Limits: one run per design, short
     learning from given positions, medium size.
   - **Second round (15 November 2026): second runs of the best five, and direct matches against D5 (first
     run). 236 games each, 200 simulations, about 2.5 points uncertain:**

     | Design | Against D0, first run | Against D0, second run | Against D5, first run | Against D5, second run |
     |---|---|---|---|---|
     | D5 line layers | 70.6% | 71.0% | | |
     | C1 = D1 + D5 | 69.1% | 65.9% | 44.5% | 43.0% |
     | C2 = D1 + D4 + D5 + D6 | 66.7% | 66.1% | 50.8% | 45.1% |
     | D1 in-board filters | 59.7% | 56.6% | | |
     | D6 exact facts | 61.2% | 56.4% | | |
     | D0 second run | 46.0% | | 28.6% | |
     | D7 big-board body | 41.3% | | 22.0% | |

     Losses of the second runs (distance / value error): D5 0.2347 / 0.1214, C1 0.2347 / 0.1223,
     C2 0.2296 / 0.1188, D1 0.2452 / 0.1273, D6 0.2660 / 0.1280.
     Reading: D5 alone is the winner. Its gain repeats (70.6% and 71.0%), and both combinations score at or
     below 50% against it in all four direct matches (C1 clearly below), although C2 has the lowest losses.
     So the in-board filters do not help on top of the line layers and seem to hurt a little; lower loss did
     not mean stronger play here. Decision: continue with D5 alone. Next: D5 at large size on the laptop,
     then the 60-round confirmation on Colab (step 3). Still untested: D7 at equal thinking time.
   - **Where the errors are (15 November 2026, the 60,000 held-back positions; D0 and D5 = average of their
     two runs; "large" is the finished large network, which has learned from these positions, so its numbers
     are flattering). Distance / value error / same best move as the search:**

     | Kind of position | Share | D0 | D5 | Large itself |
     |---|---|---|---|---|
     | all | 100% | 0.280 / 0.135 / 64% | 0.234 / 0.125 / 68% | 0.153 / 0.100 / 75% |
     | no board decided | 44% | 0.145 / 0.090 / 74% | 0.124 / 0.089 / 77% | 0.074 / 0.087 / 84% |
     | 1-2 boards decided | 24% | 0.362 / 0.127 / 60% | 0.318 / 0.126 / 62% | 0.219 / 0.107 / 71% |
     | 3-4 boards decided | 15% | 0.495 / 0.221 / 56% | 0.433 / 0.213 / 60% | 0.315 / 0.154 / 67% |
     | 5+ boards decided | 17% | 0.324 / 0.185 / 54% | 0.220 / 0.140 / 58% | 0.123 / 0.078 / 67% |
     | sent to one board | 89% | 0.255 / 0.133 / 66% | 0.220 / 0.127 / 69% | 0.146 / 0.103 / 76% |
     | free choice of board | 11% | 0.496 / 0.147 / 53% | 0.353 / 0.112 / 58% | 0.217 / 0.075 / 67% |
     | can win a small board now | 28% | 0.409 / 0.178 / 57% | 0.336 / 0.156 / 61% | 0.218 / 0.112 / 69% |

     Reading: the line layers gain most where the whole board matters (free choice: distance -29%, value
     error -24%; 5+ boards decided: -32% and -24%) and almost nothing for the value in the opening and in the
     middle game with a forced board. The hardest positions for every network, large included, are the middle
     game with 3-4 boards decided (large's value error 0.154 there, against 0.087 in the opening) and positions
     with a free choice late in the game (large finds the search's best move in only 61% of them). The opening
     value error is the same for all (0.087 to 0.090), so that is noise in the lessons, not a design limit.
     What neither D0 nor D5 has: a summary of the whole board given to every cell (needed to compare boards
     at a free choice), and lessons about the long game beyond one number (final owner of each small board,
     win/tie/loss). Candidates for the next design round: D8 = D5 + whole-board summary (mean and maximum
     over the 9 boards, sent to every cell and to the value output); the extra lessons need new games
     (second round above).
   - **Night run on the laptop (built 15 November 2026, started by the user, results open).** One command:
     `caffeinate -i /usr/bin/python3 -u src/test_designs.py night` (settings: `NIGHT` in `src/test_designs.py`;
     started again it goes on where it stopped). About 6 to 7 hours (estimate from measured steps per second).
     1. Medium, 6,000 steps, two runs each, matches against D5: D8 (D5 + whole-board summary), D5n and D5w
        (16 and 64 numbers per line instead of 32), C3 (D5 + the learned small-board summary of D2).
     2. Large size, 6,000 steps: D0 and D5, two runs each, matches against D0. Does the gain hold at large size?
     3. Medium, 36,000 steps (6 times as long): D0 and D5, measured every 1,000 steps, match D5 against D0.
        Does the gap close with longer learning?
     4. Large size, 6,000 steps: D8, two runs, matches against D5.
     Networks of another size or length are kept apart: file names end in `_steps<n>`, match names read
     `D5@large/6000`; `models/design_tests.csv` has a new column `of_steps`. The same options work by hand:
     `size=large` and `steps=36000` behind any command.
3. **Confirm on Colab:** the winning design from scratch for 60 rounds with large's settings, compared with
   large's log at the same rounds (14% at 20, 28% at 40, 35% at 60). Stop if not clearly ahead.
   - **Built 15 November 2026, not yet run:** network variant `large_lines` (96 channels, 8 blocks, line layers =
     design D5; `NN.create_model` takes the class from `src/test_designs.py` for now), command
     `python -u src/train.py run large_lines 100`, notebook `colab_design_lines.ipynb`. It saves as
     `alphazero_large_lines.pt` next to large in Drive and uses all settings of large. Tried on the laptop with
     three tiny rounds (plays, learns, checks against medium, saves, continues). Not yet run on an Nvidia
     card, and the faster network copy does not cover this design, so it plays at the old speed.
     The user allowed 6 to 7 hours, so it runs 100 rounds instead of 60 (it ends before the simulations
     rise at round 100). Large's checks: 14% at 20, 28% at 40, 35% at 60, 43% at 80, 48% at 100.
     Estimate: 5 to 6 hours, 10 to 12 units (large's first 100 rounds took 3.9 hours; the line layers are
     slower per position by a guessed 20 to 50%). 512 games per round as large had, so the comparison is
     clean; 1024 games per round is left for a later run. 
4. **The real run** with what passed, on the remaining units (about 40).

### 5. Other levers, if size stops helping

- More simulations in training late in the run (for example 800), since better search results are better lessons. Costs twice the time per round.
- A search setting between 1.0 and 1.5 in training. Untested; needs two trainings to compare.
- More rounds at the last learning rate. Medium gained only about 40 Elo from 108 extra rounds, so expect little.
- Weight decay and the other search constant (fpu) were never tuned.

### What to keep in mind throughout

- Pure network: minimax only as a measuring opponent.
- Judge by games, not by the loss.
- Change one thing at a time where possible, so its effect can be read from the checks.
- Units are charged while a session is connected, whatever it does; bought units expire (check the date in Colab).

## What others have built (web research, 15 November 2026)

- **uttt.ai** (github.com/ar-nowaczynski/utttai): the best-known public network AI. About 5 million weights, 256 channels, only 3 blocks. Its first layer looks at each small board as one unit (3x3 step 3), so the body works on a 3x3 grid of small boards. Trained in only two stages of about 8 million positions each: first from plain MCTS games with random playouts, then from its own games. Many simulations per position ("quality over quantity"). With 1,000 simulations it beats plain MCTS with 10 million. Author says a third stage was never tried.
- **CodinGame arena** (100 ms per move, one weak processor, no graphics card): the top bots use tiny networks of about 215,000 weights that only judge positions (no move suggestions), with a fast best-first search, trained on over 300 million positions. Their rules break ties by small boards won, so their numbers on first-player wins (73% for X in one author's self-play) do not carry over to our rules.
- **Solved?** Only the variant where a player sent to a decided board must still play there is proven a first-player win (2020 paper "At most 43 moves, at least 29"). Our rules are unsolved.
- **Compared with us:** large saw about 6.4 million positions in 240 rounds (each used 4 times). Others used similar or far larger amounts of data, and a design built around the small boards rather than more blocks. Ideas this suggests: more and better data before more size; a first layer that takes each small board as one unit; an extra output for the value of each move.

## Match against uttt.ai (laptop, 15 November 2026)

Medium round 240 against uttt.ai's stage-2 network, 236 games (118 different openings, both sides),
equal simulations for both. Their files and the match script are local only (`external/`, `src/test_vs_utttai.py`, both ignored by git);
results in `models/alphazero_medium_vs_utttai.csv`.

| Simulations | Their network uses | Medium's points | Wins-ties-losses |
|---|---|---|---|
| 500 | this project's search | 47.0% (+-2.3) | 54-114-68 |
| 500 | its own settings (exploration 2.0, untried move counts as even, prior at least 1%) | 51.5% (+-2.2) | 59-125-52 |
| 2000 | its own settings | 57.8% (+-2.0) | 65-143-28 |
| 2000 | this project's search | 50.6% (+-2.0) | 47-145-44 |

Reading: with the same search for both, the two networks are level (47.0% and 50.6%). Their network plays worse with its own search settings than with this project's (medium gets 51.5% and 57.8% then), so this project's search settings are the better ones, clearly so at 2000 simulations. By who moved first, over all games: at 500 simulations X won 30%, O 19%, ties 51%; at 2000 simulations X 23%, O 16%, ties 61%. Their network is about 2.2 times slower per position
than medium on the Mac (about as slow as large), so equal simulations favours them in time. In 55% of the
opening pairs exactly one of the two games was decided, so most results come from mistakes in play, not from the opening.

## Ideas for a network that fits the game better (15 November 2026, none tested)

The 3x3 filters on the 9x9 picture only fit the centre cell of a small board; 44% of what they look at lies
across a small-board border, where neighbourhood means nothing. Ideas, in my order of preference:

1. **More answers per position** (more lessons from the same games): who ends up owning each small board,
   win/tie/loss as three chances instead of one number, the search's value of every move (uttt.ai does this).
   Needs the extra outcomes stored with the positions, so it can only be tested with new games.
2. **Whole-board layer and a permanent board track:** every cell sees exactly the 9 cells of its own small board
   (a filter that steps from board to board); each small board keeps a wide description through the whole
   network instead of an average that is recomputed and dropped in every block; the value is read from that track.
   By count about half the work of today's block, not measured.
3. **Exact facts as input:** stones in decided boards blanked out, cells that win a small board at once (for
   either side), small boards whose win would win the game, lines that are still open for each side.
4. **Line layers:** the 8 lines of a small board, and the 8 lines of the big board, as built-in units with shared weights.
5. Lower priority: the reverse of the sending rule (from which cells a board can be reached), a learned
   description for each of the 19,683 fillings of a small board, an attention network with the game's links built in.

Ideas 2 to 4 can be screened on the laptop on medium's saved positions (same data and steps, compare the
losses), as was done for medium against large; that test predicted large's small gain correctly.

## Open points

- Snapshot test on Colab for rounds 200 and 240 (40 to 160 are done).
- After round 240: tests on the laptop against minimax and medium at 500 and 2000 simulations.
- If large clearly beats medium: the bigger network, with games that carry over between rounds and a measured number of games at once (1024 or 2048), using the remaining units.
- Find which layer overflows in half precision (optional).
