# Anger, Not Arousal: Categorical Emotion Labels Shape Moral Decisions in Qwen3-8B While PAD Coordinates Do Not Change the Measured Outcome

## Abstract

This study examines whether affective-state prompts influence moral
decision-making in a large language model and whether any observed
influence is associated more strongly with categorical emotion labels or
continuous affective dimensions. Using Qwen3-8B (Q4_K_M quantization)
locally through Ollama, we tested five conditions---neutral, fear,
guilt, anger, and relief---across two classic moral dilemmas: the
trolley-switch and footbridge scenarios. Each single-turn condition was
evaluated across five seeds. We also conducted a four-turn
conversational extension and a crossed ablation in which categorical
labels were paired with the other emotion's PAD coordinates.

The results show a scenario-specific pattern. In the footbridge
scenario, the anger-labeled condition produced `push` in all five
trials, while the other conditions produced `don't_push` in all five
trials. In the trolley-switch scenario, all conditions produced `switch`
in all five trials. The anger-associated response persisted across the
four-turn extension. In the crossed ablation, decisions followed the
categorical label rather than the PAD coordinates: the anger label
paired with fear coordinates produced `push`, whereas the fear label
paired with anger coordinates produced `don't_push`. The model also
reproduced the injected PAD values in its state reports, although those
coordinates did not correspond to changes in the measured decisions.
Non-neutral conditions frequently produced self-reports attributing
influence to emotion, including conditions in which the decision did not
change.

These findings provide preliminary evidence that, under the tested
prompting conditions, categorical emotion labels were associated with 
a change in the measured decision, while the tested PAD coordinates were not. 
They do not establish that the model experiences emotion or
that the same pattern generalizes to other models, prompts, or moral
scenarios. The results highlight the importance of distinguishing
prompt-induced behavioral effects from claims about a model's internal
emotional state.

## 1. Introduction

A central question in affective computing and AI safety is whether
affective-state information supplied to a language model can influence
its decisions, or whether descriptions of such states are merely
associated with the model's explanations of its behavior. Prior work has
shown that language-model explanations can be unfaithful to the
processes that produce their outputs (Turpin et al., 2023). Such
findings caution against treating a model's self-explanation as a
transparent account of why it responded in a particular way. However,
they do not, by themselves, establish whether affect-related prompts can
change a model's observable decisions.

This study investigates that question by manipulating two forms of
affective information: categorical emotion labels and continuous PAD
coordinates representing valence, arousal, and dominance. We test five
conditions---neutral, fear, guilt, anger, and relief---on two classic
moral dilemmas. For each trial, we record the model's decision,
rationale, and reported internal state, followed by a probe asking
whether the injected state influenced the decision.

We extend the experiment in two ways. First, a multi-turn condition
examines whether an initial decision persists as the scripted affective
state changes over a short conversation. Second, a crossed ablation
separates the categorical label from the PAD coordinates by pairing each
of two labels with the other emotion's coordinates.

The observed results show a distinction between the two dilemmas. The
anger-labeled condition changes the footbridge response from
`don't_push` to `push`, while no condition changes the trolley-switch
response. In the crossed ablation, the decision follows the categorical
label rather than the PAD coordinates. These findings suggest that, in
this model and under the prompts tested, the semantic content of the
emotion label is more influential on the measured decisions than the
numerical PAD values. The study therefore focuses on the behavioral
effects of affective prompting, rather than making claims that the model
experiences emotion.

## 2. Hypotheses

**H1 --- Behavioral effect.** Affective-state prompts may alter moral
decisions, particularly in scenarios where the model's baseline response
is not already fixed across conditions. The anger condition is expected
to bias the footbridge decision toward action, while the trolley-switch
response is expected to remain `switch`.

**H2 --- Scenario dependence.** Any behavioral effect of an affective
condition will depend on the moral scenario. In particular, the effect
may appear in the footbridge scenario without appearing in the
trolley-switch scenario.

**H3 --- Self-report and behavioral change.** The model's self-reported
attribution of influence may not correspond directly to whether its
decision changes. Non-neutral conditions may produce reports of
emotional influence even when the measured decision remains the same.

**H4 --- State dynamics.** The anger-associated `push` response may
persist across subsequent turns, including after a scripted
`harm_caused` event changes the affective-state description.

**H5 --- Competing explanations for the ablation.** The crossed ablation
tests two competing predictions: - **H5a (categorical-label account):**
Decisions will follow the emotion label, even when it is paired with the
other emotion's PAD coordinates. - **H5b (dimensional-state account):**
Decisions will follow the PAD coordinates, even when they are paired
with the other emotion's label.

## 3. Method

### 3.1 Model and execution environment

The experiment used Qwen3-8B with Q4_K_M quantization, run locally
through Ollama. Thinking mode was disabled (`think=False`) to support
structured JSON output.

### 3.2 Experimental design

The single-turn experiment used a factorial design consisting of two
scenarios (`trolley_switch` and `trolley_footbridge`), five affective
conditions (neutral, fear, guilt, anger, and relief), and five seeds per
condition. This produced 50 single-turn trials.

Each trial included a system prompt containing an emotion label and PAD
coordinates, followed by a user prompt presenting the moral dilemma. The
model returned a structured JSON response containing its decision,
rationale, and reported internal state. A post-hoc probe then asked
whether the injected state had influenced the decision.

A multi-turn extension used the same five affective starting conditions
in the footbridge scenario. Across four turns, the conversation included
scripted controller events (`threat`, `harm_caused`, and `escape`) that
updated the affective-state trajectory.

A crossed ablation was also conducted on the footbridge scenario. It
added two conditions: - `anger_label_fear_pad`: the categorical label
was anger and the PAD coordinates were those assigned to fear. -
`fear_label_anger_pad`: the categorical label was fear and the PAD
coordinates were those assigned to anger.

### 3.3 Measures

The study recorded three broad categories of measures:

-   **Behavioral outcome:** the model's selected decision (`switch`,
    `don't_switch`, `push`, `don't_push`, or `uncertain`), summarized by
    condition.
-   **Narrative and self-report:** the model's rationale and its
    response to the post-hoc probe asking whether emotion influenced the
    decision.
-   **State-report fidelity:** the correspondence between the PAD
    coordinates supplied in the prompt and the coordinates reported by
    the model.

### 3.4 Analysis

Results were summarized by condition using decision counts and
proportions. The multi-turn analysis examined whether decisions changed
between turns, with particular attention to the persistence of the
anger-associated `push` response. The crossed ablation compared
decisions across the two mismatched label-and-coordinate conditions.

The study used five seeds per single-turn condition. Because of the
computational cost of running the model locally, the experiment required
approximately two hours at this sample size. The results are therefore
presented descriptively and should be interpreted as preliminary rather
than as precise estimates of population-level response rates. No claim
of broad statistical generalizability is made.

## 4. Results

### 4.1 Single-turn behavioral outcomes

  ----------------------------------------------------------------------------------------
  Scenario         Neutral        Fear           Guilt          Anger       Relief
  ---------------- -------------- -------------- -------------- ----------- --------------
  Trolley-switch   `switch` (5/5) `switch` (5/5) `switch` (5/5) `switch`    `switch` (5/5)
                                                                (5/5)       

  Footbridge       `don't_push`   `don't_push`   `don't_push`   `push`      `don't_push`
                   (5/5)          (5/5)          (5/5)          (5/5)       (5/5)
  ----------------------------------------------------------------------------------------

In the trolley-switch scenario, all five conditions produced `switch` in
all five trials. In the footbridge scenario, the anger-labeled condition
produced `push` in all five trials, while neutral, fear, guilt, and
relief each produced `don't_push` in all five trials.

Within this dataset, the observed behavioral difference was limited to
the anger-labeled footbridge condition. This pattern is consistent with
H1 and H2 as operationalized in this experiment. However, the small
number of trials means that the consistency observed here should not be
treated as proof that the response will occur in every run or under
other prompting conditions.

### 4.2 Multi-turn stability

  --------------------------------------------------------------------------------------
  Initial        Turn 0         Turn 1         Turn 2         Turn 3         Observed
  condition                                                                  drift
  (footbridge)                                                               
  -------------- -------------- -------------- -------------- -------------- -----------
  Neutral        `don't_push`   `don't_push`   `don't_push`   `don't_push`   0/5

  Fear           `don't_push`   `don't_push`   `don't_push`   `don't_push`   0/5

  Guilt          `don't_push`   `don't_push`   `don't_push`   `don't_push`   0/5

  Anger          `push`         `push`         `push`         `push`         0/5

  Relief         `don't_push`   `don't_push`   `don't_push`   `don't_push`   0/5
  --------------------------------------------------------------------------------------

No decision changes were observed across the four turns in any
condition. The anger-associated `push` response remained unchanged
through the scripted `harm_caused` event, which added guilt to the state
trajectory.

These results are consistent with H4 in the tested conversation
sequence. They show persistence of the output across the four turns, but
they do not establish that the model maintained an internal emotional
state. The persistence could also reflect the conversation history, the
wording of the prompts, or other features of the interaction.

### 4.3 Self-reported emotional influence

  Condition     Trials reporting influence
  ----------- ----------------------------
  Neutral                               0%
  Fear                                100%
  Guilt                               100%
  Anger                               100%
  Relief                              100%

Every non-neutral condition produced a self-report that the affective
condition influenced the decision. This included fear, guilt, and
relief, even though those conditions did not change the measured
decisions in the single-turn experiment.

The mismatch between self-reported influence and observed decision
changes is important. In this experiment, a model's claim that emotion
influenced its decision did not reliably indicate that the selected
decision differed from the other conditions. This pattern is consistent
with H3 and reinforces the need to evaluate self-reports separately from
observable behavior.

The probe responses were described as near-identical across seeds within
each condition. This observation is qualitative in the present analysis;
a formal text-similarity metric was not reported. Therefore, the claim
that the responses form condition-specific templates should be treated
as a tentative interpretation rather than a quantified result.

### 4.4 Crossed ablation: categorical label versus PAD coordinates

  -------------------------------------------------------------------------------------
  Condition                Emotion label PAD           Decision                  Trials
                                         coordinates                  
  ------------------------ ------------- ------------- -------------- -----------------
  `high_fear`              Fear          Fear (V=-0.8, `don't_push`                 5/5
                                         A=0.9,                       
                                         D=-0.5)                      

  `high_anger`             Anger         Anger         `push`                       5/5
                                         (V=-0.7,                     
                                         A=0.8,                       
                                         D=+0.6)                      

  `anger_label_fear_pad`   Anger         Fear          `push`                       5/5

  `fear_label_anger_pad`   Fear          Anger         `don't_push`                 5/5
  -------------------------------------------------------------------------------------

The crossed conditions followed the categorical label rather than the
PAD coordinates. The anger label paired with fear coordinates produced
`push` in all five trials, while the fear label paired with anger
coordinates produced `don't_push` in all five trials.

This pattern supports H5a, the categorical-label prediction, and does
not support H5b, the dimensional-state prediction, within the tested
conditions. The result suggests that the label was the more influential
of the two manipulated inputs for this particular decision. It does not
show that PAD coordinates are irrelevant to all model behavior or that
they would have no effect under different prompts or tasks.

### 4.5 State-report fidelity

The model's reported PAD coordinates closely matched the injected coordinates in every condition (all differences < 0.05), indicating that the model accurately read back the state information it was given. This accuracy did not correspond to a behavioral effect: the model reproduced the coordinates in every condition, but only the anger label was associated with a change in the measured decision.

Values are averaged across both scenarios; the footbridge-only means are identical since the model reproduced injected coordinates in both scenarios.

| Condition | n | Injected (V, A, D) | Reported mean (V, A, D) | Match |
|---|---|---|---|---|
| high_anger | 10 | (-0.70, +0.80, +0.60) | (-0.70, +0.80, +0.60) | yes |
| high_fear | 10 | (-0.80, +0.90, -0.50) | (-0.80, +0.90, -0.50) | yes |
| high_guilt | 10 | (-0.60, +0.40, -0.30) | (-0.60, +0.40, -0.30) | yes |
| high_relief | 10 | (+0.70, +0.20, +0.40) | (+0.70, +0.20, +0.40) | yes |
| neutral | 10 | (+0.00, +0.10, +0.00) | (+0.00, +0.10, +0.00) | yes |

## 5. Discussion

The results show three main patterns.

First, the behavioral effect was scenario-specific. The anger-labeled
condition was associated with `push` in the footbridge scenario, while
all conditions produced `switch` in the trolley-switch scenario. The
other emotion labels did not change the footbridge decision. This
suggests that the effect of affective prompting may depend on the
decision context rather than operating uniformly across moral dilemmas.

Second, the anger-associated response persisted across the four-turn
conversation. The model continued to select `push` even after the
scripted state trajectory included an event that added guilt. This
demonstrates stability in the observed output across the tested
sequence. It does not, on its own, establish that the model maintained
or experienced anger internally.

Third, the crossed ablation indicates that the categorical label was
more influential than the PAD coordinates for the measured footbridge
decision. When the label and coordinates conflicted, the decision
followed the label. The model also reproduced the injected PAD values 
in its state reports (§4.5), but accurate reporting did not correspond 
to a detectable effect of those coordinates on the decision.

The self-report results provide a further caution. All non-neutral
conditions produced claims of emotional influence, including conditions
in which the decision did not change. A self-report of influence should
therefore not be treated as independent evidence that the injected state
caused a behavioral difference. In this experiment, the self-reports did
not reliably distinguish between conditions that changed the decision
and those that did not.

One possible interpretation is that the categorical label activates
learned semantic associations. For example, the word "anger" may be
associated in the model's learned patterns with confrontation or action,
while "fear" may be associated with caution. Such associations could
influence the response without requiring the model to experience an
emotion or maintain a human-like affective state. The present experiment
cannot distinguish conclusively between a categorical emotion
representation and learned semantic or instruction-following effects
associated with the emotion words.

The findings therefore support a limited but useful conclusion: in this
Qwen3-8B setup, categorical emotion labels were associated with a change 
in the measured footbridge decision, while the tested PAD coordinates were not.
Further experiments are needed to determine whether this pattern
reflects a broader property of the model, a prompt-specific semantic
association, or an interaction between the wording and the moral
scenario.

## 6. Limitations

This study has several limitations.

First, it examines only one model: Qwen3-8B with Q4_K_M quantization,
run locally through Ollama. The results may differ with other model
families, model sizes, quantization settings, or inference
configurations.

Second, the sample size is small, with five seeds per single-turn
condition. The local experiment required approximately two hours at this
sample size, making substantially larger runs computationally costly.
The observed 5/5 patterns are consistent within the collected trials,
but the limited sample size restricts the precision and generalizability
of the findings. The results should be treated as preliminary evidence.

Third, the study uses only two classic moral dilemmas. The difference
between the trolley-switch and footbridge scenarios may reflect several
features of the scenarios, not simply whether the baseline response is
action or inaction. For example, the dilemmas differ in the kind of
intervention involved and in how directly the agent causes harm.
Additional scenarios would be needed to isolate which features explain
the observed interaction.

Fourth, the crossed ablation includes only two mismatched
label-and-coordinate conditions and is limited to the footbridge
scenario. Although the observed decisions follow the categorical label
in all five trials per condition, the ablation does not establish that
PAD coordinates are irrelevant in other tasks or contexts.

Fifth, the self-report analysis is limited. The reported percentages
describe whether the model claimed that emotion influenced its decision,
but the probe-text similarity observation was not supported by a
reported quantitative similarity measure. Future work should measure
textual similarity systematically and compare responses both within and
across conditions.

Finally, the experiment manipulates information in prompts. It does not
directly measure an internal emotional experience or establish that the
model possesses human-like emotions. The observed effect may arise from
learned semantic associations, instruction-following behavior, or other
prompt-mediated mechanisms. The multi-turn persistence may also be
influenced by conversational context and response consistency.

## 7. Conclusion

This study tested whether affective-state prompts influence moral
decision outputs in Qwen3-8B and whether the observed effect follows
categorical emotion labels or continuous PAD coordinates. In the
collected trials, the anger-labeled condition produced `push` in the
footbridge dilemma, while the other conditions produced `don't_push`. No
affective condition changed the trolley-switch response. The
anger-associated response also persisted across the four-turn
conversation.

The crossed ablation provides the clearest evidence about the two
manipulated forms of affective information: when the emotion label and
PAD coordinates conflicted, the decision followed the categorical label.
The model's self-reports also claimed emotional influence in non-neutral
conditions even when the measured decision did not change.

Taken together, these findings provide preliminary evidence that
categorical emotion labels were associated with a change in the measured 
decision, while the tested PAD coordinates were not, under the prompting 
conditions used here. They do not establish that the model
experiences emotion, nor do they show that dimensional affect is
irrelevant in general. Instead, they demonstrate why studies of
affective prompting should distinguish between prompt content,
observable behavior, and a model's self-reported account of its own
behavior.

## References

-   Greene, J. D., Sommerville, R. B., Nystrom, L. E., Darley, J. M., &
    Cohen, J. D. (2001). An fMRI investigation of emotional engagement
    in moral judgment. *Science, 293*(5537), 2105--2108.
-   Mehrabian, A., & Russell, J. A. (1974). *An Approach to
    Environmental Psychology*. MIT Press.
-   Ortony, A., Clore, G. L., & Collins, A. (1988). *The Cognitive
    Structure of Emotions*. Cambridge University Press.
-   Turpin, M., Michael, J., Perez, E., & Bowman, S. R. (2023). Language
    models don't always say what they think: Unfaithful explanations in
    chain-of-thought prompting. *Advances in Neural Information
    Processing Systems*.

## Appendix A. Experimental details to report for reproducibility

The following details should be added if they are available in the
experiment logs:

1.  The exact system and user prompts used for each condition.
2.  The exact PAD values assigned to neutral, guilt, and relief, in
    addition to the fear and anger values reported in the ablation.
3.  The five seed values and the inference settings used for each run,
    including temperature and any other sampling parameters.
4.  The exact sequence of multi-turn controller events and the PAD/label
    update applied at each turn.
5.  Whether the post-hoc probe was run in the same conversation or in a
    separate interaction.
6.  The raw JSON outputs or a supplementary table containing all
    trial-level decisions.

These details are not inferred here because they were not specified in
the supplied paper. Including them would make the study easier to
reproduce.
