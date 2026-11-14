# Training notes: the large network on Google Colab

Last updated: 14 November 2026, at round 116 of the large network.
This file is the memory of the project's training decisions. It is kept up to date
whenever something that influences the training changes.

## Where things stand

- The **large** network (96 channels, 8 blocks, 1.84 million weights) trains from scratch on Colab, on the L4 card.
- From round 76 to 116 it stayed flat at about **46%** against the finished medium network (round 240), within the noise of the check (about 2 to 3 points up or down).
- The training ends by itself after round 240 (`ITERATIONS` in `src/train.py`). Then decide about an even bigger network with the remaining compute units.

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

Reading at round 116: ten checks in a row (rounds 80 to 116) average 45.9% with no trend, so the
learning rate of 0.001 has stopped giving strength. The loss fell after the step to 400 simulations
(1.293 to 1.263), as expected from sharper targets, without a gain in the check.

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

## Known problems

- **Half precision fails in rare positions** (13 of 249,000 at round 32, more later): a number inside the network gets too big and the answer is "not a number". This ruined the network in round 33. Now those positions are asked again in full precision, damaged positions never reach the memory, and learning stops without saving if the loss is not a number. Which layer overflows is not known.
- **Rounds went from 105 to about 120 seconds** after round 37; cause not found; accepted.
- **The stop button on Colab can send its signal twice.** Saving now ignores further signals. Wait for "Saved to ...".
- **A dropped session** loses at most the running round; the memory can be up to 4 rounds older.

## How to

- **Train:** `colab_train.ipynb`, L4, run the cells from the top. After a disconnect run all cells again.
- **Use new code:** stop the training cell, run the code cell, run the training cell.
- **Go back to a snapshot:** `!python -u src/train.py restore <round>`
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

### 1. Finish the large network (now)

- Let it run to round 240 even if it passes 50% against medium early. How far above medium it ends decides the next step.
- Done at round 116: the checks were flat, the learning-rate step at round 120 stays.
- Next look at about round 140 to 150: the lower learning rate should lift the points within 10 to 25 rounds. If they are still at or below 48%, large is not clearly better than medium at 100 simulations and the snapshot test at 500 and 2000 simulations decides.
- The step from 200 to 400 simulations at round 100 lowers the policy loss by itself (sharper targets), so compare losses only within the same number of simulations.

### 2. Measure the finished large network

- On Colab: the snapshot test (`colab_test.ipynb`) for all multiples of 40, at 500 and 2000 simulations.
- On the laptop: the final network against minimax on the saved openings, as for medium.
- The check during training uses only 100 simulations; the snapshot test is the firmer measure.

### 3. Decide about a bigger network

| Final result of large against medium (500 and 2000 simulations) | Meaning | What to do |
|---|---|---|
| 65% or more | size helps a lot | train the bigger network |
| about 55 to 65% | size helps somewhat | bigger is reasonable, expect a smaller gain |
| about 50 to 55% | little gain from size | do not go bigger; look at other levers first |
| still rising at round 240 | not finished | train large further (raise `ITERATIONS`) before deciding |

### 4. If the bigger network is trained (`xlarge`: 128 channels, 10 blocks, about 4 million weights)

- **Cost:** about 1.75 times large per round on the T4; measure on the L4 first (`pretest.py speed xlarge`). With about 65 units left, about 240 rounds should fit, but check the estimate before starting.
- **Use the card better:** let games carry over from one round to the next, so 1024 or more run at once without making the rounds bigger. This needs a change to the self-play code and a test first. Measure 1024 and 2048 at once with the speed test.
- **Half precision:** expect the overflow problem again, probably earlier. The repair is in place; finding the layer that overflows and running only that layer in full precision would remove the cause.
- **Schedule:** do not copy round numbers. Use the same rule as for large: keep the learning rate while the points against the reference keep rising, lower it when they go flat.
- **Reference network:** the check should then be against the finished large network (set `REFERENCE = 'large'`), with medium as a second line if wanted.
- **Compare with large's log** at the same rounds while the settings are the same; if the bigger network is not ahead of large by about round 40 to 60, stop and save the units.
- **Playing speed on the laptop:** the bigger network needs about twice the time per move of large.

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

## Open points

- Snapshot test on Colab for rounds 40 and 80, later 120, 160, 200, 240.
- After round 240: tests on the laptop against minimax and medium at 500 and 2000 simulations.
- If large clearly beats medium: the bigger network, with games that carry over between rounds and a measured number of games at once (1024 or 2048), using the remaining units.
- Find which layer overflows in half precision (optional).
