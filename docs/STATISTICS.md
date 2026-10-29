# Statistical Definitions

Each paired comparison uses the same ten seeds 10,20,...,100. Directions are
specified by `contrast` and the original comparison identifier. Positive
differences do not always favor the same method; interpret the named contrast.

For paired differences d, the corrected standard error is
`sqrt(1/10 + 1/8) * sample_sd(d) = sqrt(.225) * sample_sd(d)`.
Individual 95% intervals use the t distribution with 9 degrees of freedom.
Zero-variance contrasts retain raw p=1 and an undefined interval.

The complete original adjustment families are Holm184, Holm100, Holm40,
Holm16, Holm64, Holm138 and Holm90. They address different analyses; they are
not a single global family. All632 comparisons, including unfavorable ones,
are retained. An individual interval excluding zero can coexist with an
adjusted p above .05 because the interval is not simultaneous.

The restricted-search family includes four contrasts for every one of 16 cells:
baseline9 minus conditioned9, conditioned9 minus full180, independently tuned
Risk-only minus full180, and independently tuned Risk-only minus conditioned9.
Fixing lambda=.5 limits candidate count, not model capacity or computation.

Risk-only comparisons distinguish the independently tuned competitor from the
inherited-setting component ablation. Shared Soft Gate contexts do not become
independent repetitions. Random costs are averaged within seed before inference.

The ten holdouts overlap. HMEQ Random/Shuffled follow-up comparisons reuse
previously examined data and remain exploratory. Geometry correlations are
descriptive associations across these overlapping splits. Matched shuffled
Gram matrices have no classifier-cost outcomes.
