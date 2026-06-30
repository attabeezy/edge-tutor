# Calculus Made Easy — Local Arctic ONNX Routing

- Chunks: `141`
- Index build/load time: `0.06s`
- Threshold separation: `FAIL`

## Held-out metrics

- Grounded recall: `0.0%`
- False grounding rate: `0.0%`
- General-task precision: `100.0%`
- Clarification rate: `0.0%`

When threshold separation fails, the metrics above use `UNRESOLVED` and are not release scores. The held-out scalar trade-off is:

- Preserve ≥95% grounded recall: threshold `0.602162`, recall `100.0%`, false grounding `80.0%`.
- Keep false grounding ≤5%: threshold `0.675850`, recall `60.0%`, false grounding `0.0%`.

## Calibration score ranges

- GROUNDED: `0.622265`–`0.742458` (median `0.668298`)
- UNSUPPORTED: `0.553899`–`0.622740` (median `0.604842`)

## Held-out cases

| Case | Expected | Predicted | Mean top-5 | Top passage |
|---|---|---|---:|---|
| t-g01 | GROUNDED | UNRESOLVED | 0.6022 | bol, you will henceforth know that it is put there merely to give you instructions that you are now to perform |
| t-g02 | GROUNDED | UNRESOLVED | 0.6484 | a similar THE LAW OF ORGANIC GROWTH 155 way. In fact ϵ−atserves as a die-away factor for all those phenomena i |
| t-g03 | GROUNDED | UNRESOLVED | 0.6995 | EASY 140 Now we might have gone to work the other way, and said: Go to; let us find a function of x, such that |
| t-g04 | GROUNDED | UNRESOLVED | 0.6612 | EASY 140 Now we might have gone to work the other way, and said: Go to; let us find a function of x, such that |
| t-g05 | GROUNDED | UNRESOLVED | 0.6759 | with respect to y. Also ϵax, which is equal to ( ϵa)x, will, when differentiated with respect to x, beaϵax, be |
| t-g06 | GROUNDED | UNRESOLVED | 0.6874 | EASY 140 Now we might have gone to work the other way, and said: Go to; let us find a function of x, such that |
| t-g07 | GROUNDED | UNRESOLVED | 0.7043 | the area of which is twice the area of a polar diagram, is equal to the quadratic mean of all the values ofrfo |
| t-g08 | GROUNDED | UNRESOLVED | 0.7208 | at any point is dy dx= 4x. For the point where x= 0, this slope is zero; the curve is horizontal. For the poin |
| t-f01 | GROUNDED | UNRESOLVED | 0.7098 | of variation (or “fluxion”) was ˙ y. Ifxwas the variable, then its fluxion was called ˙ x. The dot over the le |
| t-f02 | GROUNDED | UNRESOLVED | 0.6479 | must remember that what we have got to sum up together is not all the dx’s, but all such terms as x2dx; and th |
| t-u01 | UNSUPPORTED | UNRESOLVED | 0.6051 | bol, you will henceforth know that it is put there merely to give you instructions that you are now to perform |
| t-u02 | UNSUPPORTED | UNRESOLVED | 0.6666 | CALCULUS MADE EASY 230 indeed the equation is seen to possess some standard form of which the integral is know |
| t-u03 | UNSUPPORTED | UNRESOLVED | 0.6434 | 3)2. (12) ax(axa−1+xalogϵa). (14) Min.: y= 0.7 for x= 0.694. (15)1 +x x. (16)3 x(logϵax)2. Exercises XIII. (p. |
| t-u04 | UNSUPPORTED | UNRESOLVED | 0.6253 | only functions of which the second differ- ential coefficient is equal (and of opposite sign to) the original  |
| t-u05 | UNSUPPORTED | UNRESOLVED | 0.6566 | we should get dyv=v du; or if we treat uas a constant, and differentiate with respect to v, we should have: dy |
| t-u06 | UNSUPPORTED | UNRESOLVED | 0.6136 | that what one fool can do, other fools can do also , it lets you see that these mathematical swells, who pride |
| t-u07 | UNSUPPORTED | UNRESOLVED | 0.6324 | part of the curve y=x−x2, which is shown in Fig. 60. To find the mean ordinate, we NM 11/4 OY Fig. 60. shall h |
| t-u08 | UNSUPPORTED | UNRESOLVED | 0.6179 | CALCULUS MADE EASY 130 It follows thatdy dx=−1 3p (θ+ 5)4, as might have been found oth- erwise. We shall find |
| t-u09 | UNSUPPORTED | UNRESOLVED | 0.5373 | that what one fool can do, other fools can do also , it lets you see that these mathematical swells, who pride |
| t-u10 | UNSUPPORTED | UNRESOLVED | 0.4865 | as a,b, orc; while those which we consider as capable of growing, or (as mathematicians say) of “varying,” we  |
| t-a01 | GENERAL_TASK | GENERAL_TASK | 0.5838 | the amount you spend in a short CALCULUS MADE EASY 54 time dtbe called dy, the rateof spending it will bedy dt |
| t-a02 | GENERAL_TASK | GENERAL_TASK | 0.6328 | bol, you will henceforth know that it is put there merely to give you instructions that you are now to perform |
| t-a03 | GENERAL_TASK | GENERAL_TASK | 0.5803 | as a,b, orc; while those which we consider as capable of growing, or (as mathematicians say) of “varying,” we  |
| t-a04 | GENERAL_TASK | GENERAL_TASK | 0.6401 | to the yfrom which it was derived. The contrast between the two processes may be illustrated by the following  |
