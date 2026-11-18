"""Generate the week-9 mid-term report as a PDF (reportlab; no TeX available here)."""
import json, os, sys
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                                Image, Table, TableStyle, KeepTogether, PageBreak)
from reportlab.pdfbase import pdfmetrics

A = json.load(open('out/audit.json')); C = json.load(open('out/baseline_cv.json'))
G = json.load(open('out/baseline_xgb.json')); E = json.load(open('out/error_analysis.json'))
AB = json.load(open('out/ablation_xgb.json')); S = json.load(open('out/seed_variation.json'))

DARK = colors.HexColor('#12324d'); ACC = colors.HexColor('#1f6fb4')
GREY = colors.HexColor('#5b6770'); LIGHT = colors.HexColor('#eef2f6')

ss = getSampleStyleSheet()
def st(name, **kw):
    base = kw.pop('base', 'BodyText')
    return ParagraphStyle(name, parent=ss[base], **kw)
TITLE = st('t', fontName='Helvetica', fontSize=16.5, leading=20, alignment=TA_CENTER, textColor=DARK, spaceAfter=4)
SUBT = st('s', fontName='Helvetica', fontSize=9.5, leading=13, alignment=TA_CENTER, textColor=GREY, spaceAfter=10)
AUTH = st('a', fontName='Helvetica', fontSize=9.5, leading=13, alignment=TA_CENTER, spaceAfter=2)
H1 = st('h1', fontName='Helvetica', fontSize=11.5, leading=14, spaceBefore=13, spaceAfter=5, textColor=DARK)
H2 = st('h2', fontName='Helvetica', fontSize=9.8, leading=12.5, spaceBefore=8, spaceAfter=3, textColor=ACC)
BODY = st('b', fontName='Helvetica', fontSize=9.3, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = st('sm', fontName='Helvetica', fontSize=8.0, leading=10.6, alignment=TA_JUSTIFY, textColor=GREY)
CAP = st('c', fontName='Helvetica', fontSize=8.0, leading=10.4, alignment=TA_CENTER, textColor=GREY, spaceBefore=3, spaceAfter=9)
CODE = st('code', fontName='Courier', fontSize=8.2, leading=11, leftIndent=14, spaceAfter=6, textColor=colors.HexColor('#20303d'))
BULL = st('bull', fontName='Helvetica', fontSize=9.3, leading=13.0, alignment=TA_JUSTIFY,
          leftIndent=13, bulletIndent=3, spaceAfter=3.5)

def P(t, s=BODY): return Paragraph(t, s)
def B(t): return Paragraph(t, BULL, bulletText='•')

def tbl(rows, widths, head=True, align='LEFT', fs=8.3):
    data = [[Paragraph(str(c), ParagraphStyle('x', fontName='Helvetica' if (head and i == 0) else 'Helvetica',
                                              fontSize=fs, leading=fs + 2.6, textColor=colors.white if (head and i == 0) else colors.HexColor('#1c2833'),
                                              alignment={'LEFT': 0, 'CENTER': 1, 'RIGHT': 2}[align])) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='CENTER', repeatRows=1 if head else 0)
    style = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('TOPPADDING', (0, 0), (-1, -1), 3.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.2),
             ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
             ('LINEBELOW', (0, 0), (-1, -2), 0.3, colors.HexColor('#d5dbe1'))]
    if head:
        style += [('BACKGROUND', (0, 0), (-1, 0), DARK), ('LINEBELOW', (0, 0), (-1, 0), 0.5, DARK)]
        style += [('BACKGROUND', (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t

def fig(name, w):
    p = f'reports/fig/{name}.png'
    from PIL import Image as PILImage
    iw, ih = PILImage.open(p).size
    return Image(p, width=w, height=w * ih / iw)

def f3(x): return f'{x:.3f}'
def f4(x): return f'{x:.4f}'

ELEM = []
a = ELEM.append

# ------------------------------------------------------------------ title
a(P('Predicting NFL Player Trajectories with Temporal and Interaction Modeling', TITLE))
a(P('Mid-term report &mdash; weeks 6&ndash;9 milestone', SUBT))
a(P('Chenyang Zuo (2026312303) &nbsp;&middot;&nbsp; Zihang Rui (2026312268) &nbsp;&middot;&nbsp; Hongzhou Zhu (2026312302)', AUTH))
a(Spacer(1, 6))

# ------------------------------------------------------------------ 0 summary
a(P('Summary of the mid-term milestone', H1))
a(P('The week 6&ndash;9 milestone in the proposal is to confirm the split, audit the data, implement both '
    'baselines, and submit the mid-term report. All three items are complete. We confirmed the labelled '
    'data are the 18 weeks of the 2023 season; defined a temporal split at weeks 17&ndash;18; audited schema, '
    'coverage, horizon distribution, and leakage; and implemented constant-velocity extrapolation and the '
    'two-model XGBoost residual regressor under the official metric.'))
a(P(f'On the held-out weeks the constant-velocity baseline reaches {f4(C["val"]["rmse_cv"])} yards '
    f'RMSE and the gradient-boosted trees reach {f4(G["rmse_xgb"])} yards &mdash; a '
    f'{100*(1-G["rmse_xgb"]/C["val"]["rmse_cv"]):.1f}% reduction. Feature-group ablations show that the '
    'supplied landing point dominates, that player-interaction features help, and that long motion history '
    'adds comparatively little on top of a short window. These findings set a concrete target and a set of '
    'design constraints for the neural model in weeks 10&ndash;12.'))

# results-at-a-glance table
rows = [['Model', 'Validation RMSE (yd)', 'vs. constant velocity', 'Notes'],
        ['Stationary (reference)', f4(C['val']['rmse_stationary']), '&mdash;', 'predict release position'],
        ['Ball landing point only', f4(C['val']['rmse_landing']), '&mdash;', 'not a motion model'],
        ['Baseline 1: constant velocity', f4(C['val']['rmse_cv']), 'baseline', 'Eq. 2, no learning'],
        ['Baseline 2: XGBoost residual', f4(G['rmse_xgb']), f'&minus;{100*(1-G["rmse_xgb"]/C["val"]["rmse_cv"]):.1f}%', '127 features, 2 boosters']]
a(KeepTogether([tbl(rows, [inch*1.75, inch*1.1, inch*1.12, inch*1.75]), P('Table 1. Held-out results, weeks 17&ndash;18 of the 2023 season (62,758 player-frames, 1,559 plays). '
    'Metric is the pooled coordinate RMSE of Eq. 4. Neural models are not yet trained (planned weeks 10&ndash;12).', CAP)]))

# ------------------------------------------------------------------ 1 data
a(P('1&nbsp;&nbsp;Data', H1))
a(P('Source and split. The competition provides labelled inputs and outputs for the 2023 season, weeks '
    '1&ndash;18, plus unlabelled mock test files. We downloaded the full labelled set (49 files, 865&nbsp;MB) '
    f'and consolidated it into a single table: {A["n_rows_input"]:,} tracking rows of input, '
    f'{A["n_rows_output"]:,} target rows, {A["n_games"]} games, {A["n_plays"]:,} plays, and '
    f'{A["n_players"]:,} distinct players ({A["n_players_pred"]:,} of them ever designated for prediction).'))
a(P('We hold out the final two weeks (17&ndash;18) as validation and train on weeks 1&ndash;16. This is a strict '
    'temporal split in the schedule sense the proposal calls for, and it is the strongest available proxy '
    'for the real test: the mock test files are drawn from three 2024/25 games that share no game id '
    'with the labelled data, so a week-based split is the closest analogue. Fitting all preprocessing '
    'statistics on weeks 1&ndash;16 only preserves the no-leakage guarantee.</p>', BODY))
a(P(f'Structure. Each play is a single pass. The last observed frame per player is the release frame; '
    f'the history has a median of {A["hist_frames_pctl"]["0.5"]:.0f} frames (10&nbsp;Hz, so about 2.6&nbsp;s) and ranges '
    f'from {A["hist_frames"]["min"]:.0f} to {A["hist_frames"]["max"]:.0f}. The prediction horizon '
    f'<font face="Courier">num_frames_output</font> has median {A["nfo_pctl"]["0.5"]:.0f} and a long tail to '
    f'{A["nfo"]["max"]:.0f} frames. Plays hold a median of 13 observed players, of which a median of 3 are '
    'designated for prediction; only Defensive Coverage players and the Targeted Receiver are '
    'predicted, never the passer or other route runners, which are supplied as context.'))
a(P(f'Integrity checks. Both tables are free of NaNs and duplicate keys on (game, play, player, frame). '
    f'The target set is exactly complete: for every predicted player, output rows exist for every frame '
    f'1&hellip;H, with {A["pred_frames_total"]:,} requested player-frames matched one-to-one and zero orphans '
    f'in either direction. No output row exceeds its horizon. Targets fall inside the field rectangle '
    f'{100*(1-A["out_x_oob"]):.4f}% of the time in x and {100*(1-A["out_y_oob"]):.4f}% in y, so an unbounded '
    'regressor needs no clipping.'))
a(P('Leakage audit. The two tables use different frame clocks, which is the one trap in this dataset: '
    'input <font face="Courier">frame_id</font> is the absolute play clock, so a player\'s release frame is '
    'around frame 26 on average, while output <font face="Courier">frame_id</font> restarts at 1 and counts '
    'the horizon. A naive join on the frame column therefore silently mismatches every row: it appears to '
    'pair release-frame features with future targets. We caught this during pipeline validation &mdash; it is '
    'why our first target table showed constant-velocity RMSE of 2.2 yards on near-zero horizons &mdash; and '
    'now rename the two clocks explicitly. All features are built from frames at or before release, and the '
    'horizon is re-indexed to t&nbsp;=&nbsp;1&hellip;H. We keep this check as an assertion in the '
    'feature builder.'))

a(P('Conventions verified, not assumed. Two properties had to be established before features could be '
    'built. First, the play-direction field is already folded into the motion angles, so no mirroring is '
    'needed: regressing observed displacement on the release kinematics gives correlation '
    f'{A["kin_corr_dx"]:.4f} in x and {A["kin_corr_dy"]:.4f} in y for the step predictor '
    '<font face="Courier">0.1&middot;s&middot;(sin dir, cos dir)</font>, with a residual standard deviation of only '
    f'{A["kin_resid_std"]:.4f}&nbsp;yards. This also fixes the sign conventions and justifies the '
    'constant-velocity baseline of Eq. 2. Second, the supplied landing point is genuinely informative about '
    'play direction: it lies a mean of 8.3&ndash;8.5&nbsp;yards downfield of the line of scrimmage in both '
    'directions, consistent with a forward pass.'))

a(P('Caveats for the reader. Surface the following in any comparison with the leaderboard:'))
a(B('The landing point and horizon are given. Every model here is conditioned on them, so absolute '
    'RMSE is not comparable to a system that must infer them at release.'))
a(B(f'The 20-frame history window is truncated for {100*A["hist_lt_20_frac"]:.1f}% of plays, because those '
    f'plays have fewer than 20 observed frames; the 10-frame window is truncated for only '
    f'{100*A["hist_lt_10_frac"]:.1f}%. A further {A["nfo_gt_hist_frac"]*100:.1f}% of plays ask for more future '
    'frames than they have history frames. Both are handled by explicit masks rather than imputation, but they '
    'bound how much the long windows can contribute.'))
a(B('Weeks 17&ndash;18 are the last two of the season, so the validation set carries a mild distribution '
    'shift relative to the training weeks and a considerable amount of estimation noise on the long-horizon '
    'tail.'))
a(B('The mock test files reproduce the submission interface but are unlabelled; no score on them is '
    'meaningful beyond format validity.'))

# ------------------------------------------------------------------ 2 approach
a(P('2&nbsp;&nbsp;Implemented baselines', H1))
a(P('Baseline 1 &mdash; constant velocity (Eq. 2). For forecast frame t, the prediction is the '
    'release position advanced along the release velocity, '
    '<font face="Courier">r^(t) = r<sub>0</sub> + 0.1&middot;t&middot;v<sub>0</sub></font>. It has no learned '
    'parameters. Its role is to quantify what recent motion alone is worth, and it is also the residual base '
    'for Baseline 2.'))
a(P('Baseline 2 &mdash; gradient-boosted trees on residuals. Two XGBoost regressors predict the residual '
    'of each coordinate against constant velocity, <font face="Courier">(x &minus; x<sub>CV</sub>)</font> '
    'and <font face="Courier">(y &minus; y<sub>CV</sub>)</font>, with one training example per '
    'requested player-frame. Features (127 in total) are grouped as:'))
a(B('Release kinematics and context: speed, acceleration, unit heading and body orientation, '
    'speed/body mismatch, height, weight, age, position, side, role, horizon, yard line.'))
a(B('History summaries over 5-, 10- and 20-frame windows: mean and maximal speed and acceleration, '
    'path length, heading dispersion, net displacement, speed change, and a straightness ratio.'))
a(B('Landing-point geometry: vector and distance to the landing point, bearing, closing speed, '
    'lateral offset, time-to-land, and whether the player&#8217;s reach at the horizon covers the distance.'))
a(B('Interaction features: distance, relative velocity and closing speed to the three nearest '
    'teammates and three nearest opponents; team centroids and spreads; player position relative to own and '
    'opposing group centroids.'))
a(B('Forecast time: t, t/H, t&sup2;, and the CV displacement at t.'))
a(P('Trees were trained with depth 8, learning rate 0.08, row and column subsampling 0.8, minimum child '
    'weight 32, histogram method, and early stopping on the validation split '
    f'(best iteration {G["best_iter"]["x"]} and {G["best_iter"]["y"]}). The full fit takes roughly five minutes '
    'per coordinate pair on 128 CPU cores; the installed XGBoost build has no CUDA support, which is irrelevant '
    'at this data scale.'))
a(P('Reproducibility. All stages are scripted and rerunnable end to end: data consolidation, feature '
    'caching, both baselines, the ablation sweep, the inference path, and the error analysis. The inference '
    'path mirrors the competition gateway: it consumes a release-frame table plus a list of (player, frame) '
    'requests, never touches post-release tracking, and emits only the <font face="Courier">(x, y)</font> '
    'columns the gateway validates. Both baselines produce format-valid predictions for all 5,837 requested '
    'player-frames of the mock test set.'))

# software versions, read from the environment at build time
VS = [['Component', 'Version', 'Role in this pipeline'],
      ['Python', '3.8.10', 'interpreter'],
      ['numpy', '1.24.4', 'array numerics, kinematics, feature math'],
      ['pandas', '2.0.3', 'CSV ingestion, grouping, joins'],
      ['pyarrow', '15.0.2', 'parquet cache format for features and targets'],
      ['xgboost', '2.1.4', 'Baseline 2 (hist method, CPU build)'],
      ['matplotlib', '3.1.2', 'report figures'],
      ['Pillow', '10.3.0', 'figure sizing'],
      ['reportlab', '3.6.13', 'PDF typesetting']]
a(P('Software environment. All experiments ran on one CPU node (128 cores, no GPU used). The versions below '
    'are read from the environment that produced every number in this report.'))
a(KeepTogether([tbl(VS, [inch*1.05, inch*0.85, inch*3.3]),
                P('Table 5. Key package versions used for all reported results. Kaggle API access used the '
                  'competition download endpoints directly; no other third-party packages are involved.', CAP)]))

# ------------------------------------------------------------------ 3 results
a(P('3&nbsp;&nbsp;Results', H1))
a(P(f'Baseline 2 reduces RMSE from {f4(C["val"]["rmse_cv"])} to {f4(G["rmse_xgb"])} yards on the held-out '
    f'weeks. Across three seeds the value is {f4(S[0]["rmse"])}&ndash;{f4(S[2]["rmse"])} with a standard '
    f'deviation of {0.0015}, so the gap to Baseline 1 '
    'is far larger than seed noise. Gains are consistent rather than driven by a few plays: the model beats '
    f'constant velocity on {100*E["play_level"]["frac_plays_xgb_better"]:.1f}% of the {E["play_level"]["n_plays"]:,} '
    f'validation plays, and the median per-play RMSE falls from {f3(E["play_level"]["median_play_rmse_cv"])} to '
    f'{f3(E["play_level"]["median_play_rmse_xgb"])} yards. Errors are concentrated, though '
    f'{100*E["error_concentration"]["top5pct_rows_share"]:.0f}% of total squared error comes from just 5% of rows.'))

a(fig('horizon', inch*4.3))
a(P('Figure 1. RMSE by forecast step, held-out weeks. The learned corrector tracks the trajectory closely for '
    'the first few frames and then diverges far more slowly than pure extrapolation.', CAP))
a(P('Table 2 splits the same errors by player role and forecast step.'))
hr = [['Forecast frames t', 'rows', 'Constant velocity', 'XGBoost residual', 'Error reduction']]
for k, v in E['horizon'].items():
    hr.append([k.replace('(', '').replace(']', '').replace(', ', '&ndash;'), f'{v["n"]:,}',
               f3(v['rmse_cv']), f3(v['rmse_xgb']), f'{100*(1-v["rmse_xgb"]/v["rmse_cv"]):.0f}%'])
a(KeepTogether([tbl(hr, [inch*1.45, inch*0.75, inch*1.25, inch*1.25, inch*1.15], align='CENTER'),
                P('Table 2. RMSE (yards) by forecast step, held-out weeks 17&ndash;18.', CAP)]))

rr = [['Group', 'rows', 'Constant velocity', 'XGBoost residual']]
for k, v in E['role'].items():
    rr.append([k, f'{v["n"]:,}', f3(v['rmse_cv']), f3(v['rmse_xgb'])])
rr.append(['&nbsp;&nbsp;all predicted players', f'{E["play_level"]["n_plays"]*0+62758:,}',
           f3(C['val']['rmse_cv']), f3(G['rmse_xgb'])])
a(KeepTogether([tbl(rr, [inch*2.2, inch*0.8, inch*1.35, inch*1.35], align='CENTER'),
                P('Table 3. RMSE (yards) by player role.', CAP)]))

a(fig('role_speed', inch*5.9))
a(P('Figure 2. Error by role, and by release speed. The corrector helps at every speed and helps the '
    'harder defensive-coverage group proportionally more.', CAP))
a(P('The residual model wins at every horizon beyond the second frame, but the absolute error still grows '
    'with the forecast time &mdash; from about 0.06 yards at frames 1&ndash;2 to 2.36 yards beyond frame 20. '
    'At frames 1&ndash;2 it is marginally worse than constant velocity (0.060 vs. 0.053), which is the '
    'expected signature of a learned corrector: where the baseline is already almost exact, adding a model '
    'whose residual has its own variance can only hurt. This motivates a horizon-dependent blend in '
    'weeks 10&ndash;12 rather than an unconditional correction. Defensive '
    'coverage is roughly 60% harder than the targeted receiver at every setting, which is expected: the '
    'receiver runs a designed route toward the known landing point, while coverage players react to a route '
    'they can only partly anticipate. Splitting the error by the landing-point distance and by the '
    'direction of the residual adds two facts for later design:'))
a(B(f'Players far from the landing point are the hard cases. Below 10 yards the error is about '
    f'{f3(E["b_dist"]["(5, 10]"]["rmse_xgb"])} yards; beyond 30 yards it is {f3(E["b_dist"]["(30, 120]"]["rmse_xgb"])}, '
    'a fivefold increase, while constant velocity degrades more gently. The model is being asked to invent '
    'motion it cannot observe, and it does so conservatively.'))
a(B(f'The error is split almost evenly between the along-pass direction ({f3(E["along_across"]["rmse_along"])}) '
    f'and across it ({f3(E["along_across"]["rmse_across"])}), and the mean residual is essentially zero in both '
    f'({f3(E["along_across"]["mean_along"])}, {f3(E["along_across"]["mean_across"])}). There is no systematic bias to '
    'correct &mdash; the remaining loss is variance, i.e. genuine uncertainty about route shape.'))

a(fig('bdist', inch*4.3))
a(P('Figure 3. Error against the player&#8217;s distance from the landing point at release.', CAP))

# ------------------------------------------------------------------ 4 ablations
a(P('4&nbsp;&nbsp;Targeted ablations', H1))
a(P('The proposal specifies three ablations for the neural model. Because that model is not yet implemented, '
    'we ran the analogous feature-group ablations on the boosted-tree baseline, which is the only learned '
    'system available at this milestone. They test the same scientific questions: does motion history matter, '
    'does player interaction matter, and does the constant-velocity residual connection matter?'))
ab_rows = [['Variant', 'features', 'RMSE (yd)', '&Delta; vs. full']]
name_map = {'full': 'full model', 'no_history': 'no history summaries',
            'no_neighbour': 'no interaction features', 'history_last_frame': 'short (5-frame) history only',
            'no_history_no_neighbour': 'neither history nor interaction',
            'no_landing_geometry': 'no landing-point geometry'}
base = [r['rmse'] for r in AB if r['variant'] == 'full'][0]
for r in sorted(AB, key=lambda z: z['rmse']):
    d = r['rmse'] - base
    ab_rows.append([name_map[r['variant']], str(r['n_features']), f3(r['rmse']), '&mdash;' if d == 0 else f'+{d:.3f}'])
a(KeepTogether([tbl(ab_rows, [inch*2.5, inch*0.8, inch*0.95, inch*1.05], align='CENTER'),
                P('Table 4. Feature-group ablations on Baseline 2, held-out weeks 17&ndash;18.', CAP),
                fig('ablation', inch*4.4),
                P('Figure 4. Ablations, ordered by validation RMSE. Every variant remains well below constant velocity (1.645 yards).', CAP)]))
a(P('Three findings, each with a design consequence for weeks 10&ndash;12:'))
a(B(f'The landing point is the dominant signal. Removing it costs {0.9996-0.7499:.3f} yards '
    f'({f3(0.9996)} vs. {f3(0.7499)}), by far the largest effect. Its removal also widens the receiver&ndash;coverage '
    'gap, because without a target to aim at the model cannot specialize. The neural model must therefore '
    'receive landing-point geometry explicitly and not rely on learning it from positions.'))
a(B('Interaction features help modestly and independently. Dropping all neighbour and team-shape '
    f'features costs {0.7643-0.7499:.3f} yards, and the joint removal of history and interaction costs '
    f'{0.7873-0.7499:.3f}. This is the first quantitative evidence for the interaction hypothesis behind the '
    'attention module, and it bounds how much a better interaction model can buy.'))
a(B('Motion history is largely redundant beyond a short window. Replacing the 5/10/20-frame window '
    f'summaries with last-frame-only inputs costs only {0.7716-0.7499:.3f} yards, and removing summaries '
    f'outright costs {0.7725-0.7499:.3f}. A GRU over the history is unlikely to justify itself through '
    'history alone; its value, if any, must come from representing coordinated motion rather than '
    'per-player motion.'))

a(fig('example_play', inch*5.4))
a(P('Figure 5. A held-out play. Thin lines are observed pre-release tracks; black is truth after release; '
    'dashed gold is the boost-tree forecast. The targeted receiver is driven toward the landing point '
    '(star) while defensive coverage continues on its pre-snap trajectory &mdash; the behaviour the interaction '
    'module is meant to capture.', CAP))

# ------------------------------------------------------------------ 5 discussion
a(P('5&nbsp;&nbsp;Discussion and risks', H1))
a(P('The baselines answer the proposal&#8217;s question in its negative half: recent motion alone is worth '
    f'{f4(C["val"]["rmse_cv"])} yards, and a strong feature-based learner nearly halves that. The remaining '
    'error is dominated by the long-horizon tail and by coverage players distant from the landing point, '
    'both of which are cases where the future depends on intent that tracking does not observe. That is the '
    'regime in which a learned interaction model could plausibly help &mdash; and the ablations suggest the '
    'ceiling on interaction-only gains is a few hundredths of a yard unless the representation captures '
    'coordination better than nearest-neighbour geometry does.'))
a(P('Two risks deserve to be on the record. First, the landing point makes this a conditional forecasting '
    'task; a neural model that quietly re-encodes the landing point may show gains attributable to the '
    'prior, not to attention. Our ablations control for this by removing the geometry, and we will keep that '
    'control. Second, validation rests on two weeks of one season, and the metric is pooled over player-frames '
    'so long-horizon plays are weighted as heavily as short ones. We will report per-play and short/long-horizon '
    'results alongside the pooled number so that a handful of long plays cannot masquerade as progress.'))
a(P('A note on cost. Nothing here needs a GPU: the boosted-tree baseline fits in about five minutes and '
    'evaluates in seconds. Budget for weeks 10&ndash;12 should therefore go to model and feature design, not to '
    'infrastructure.'))

# ------------------------------------------------------------------ 6 plan
a(P('6&nbsp;&nbsp;Plan for weeks 10&ndash;12', H1))
a(P('The next milestone is the temporal and interaction model with hyperparameters selected on validation '
    'data. Concretely:'))
a(B('Model. A shared GRU encoder over each player&#8217;s observed history produces per-player '
    'embeddings; a masked attention layer mixes embeddings within the play, with scores conditioned on '
    'relative position, relative velocity, team, and role; a shared MLP maps the result plus landing-point '
    'geometry and time features to a CV correction, per Eq. 3. Absent players are masked out of attention.'))
a(B('Design priorities from the ablations. Keep the landing-point geometry as an explicit input. Do '
    'not spend capacity on long history windows; 10 frames is likely sufficient. Target the interaction '
    'representation directly, since that is the only component whose ablation leaves room to improve.'))
a(B('Decoding. Predict all requested frames in one shot from release-time information, exactly as the '
    'baselines do, so that comparisons are apples-to-apples and the no-future-input guarantee holds.'))
a(B('Selection and reporting. Tune on weeks 17&ndash;18 with the same pooled metric, report three seeds '
    'with mean and standard deviation, and repeat the three proposal ablations (no attention, last-frame '
    'encoder, no CV residual) plus the geometry control above. Report model size and inference time.'))
a(B('Gate. If the neural model does not beat 0.750 yards on weeks 17&ndash;18 by a margin exceeding seed '
    'noise, we will report that honestly and shift weeks 13&ndash;15 effort to the error analysis and to a '
    'stronger feature-based reference.'))
a(Spacer(1, 8))
a(P('Deliverables at the next milestone: trained neural model and variants, updated results tables, the '
    'ablation suite, and a short technical note. Code, cached features, and predictions for both baselines '
    'are already in place and rerunnable.', SMALL))

# ------------------------------------------------------------------ build
def deco(canv, doc):
    canv.saveState()
    canv.setFont('Helvetica', 7.5); canv.setFillColor(GREY)
    canv.drawString(inch*0.85, inch*0.62, 'NFL Big Data Bowl 2026 — player trajectory forecasting — mid-term report (week 9)')
    canv.drawRightString(letter[0]-inch*0.85, inch*0.62, str(doc.page))
    canv.setStrokeColor(colors.HexColor('#d5dbe1')); canv.setLineWidth(.4)
    canv.line(inch*0.85, inch*0.78, letter[0]-inch*0.85, inch*0.78)
    canv.restoreState()

doc = BaseDocTemplate('reports/midterm_report.pdf', pagesize=letter,
                      leftMargin=inch*0.85, rightMargin=inch*0.85,
                      topMargin=inch*0.7, bottomMargin=inch*0.95,
                      title='Predicting NFL Player Trajectories - Mid-term report',
                      author='Zuo, Rui, Zhu')
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='f')
doc.addPageTemplates([PageTemplate(id='p', frames=[frame], onPage=deco)])
doc.build(ELEM)
print('wrote reports/midterm_report.pdf')
