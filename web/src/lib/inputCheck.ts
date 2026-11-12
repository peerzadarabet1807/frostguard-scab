/** Every sentence the leaf scan shows about the API's input check ("does this look like an apple leaf?"). */
export const INPUT_CHECK_COPY = {
  heading: "This doesn't look like an apple leaf.",
  why: "The detector was trained on apple leaves, so its answer for this photo would not mean anything.",
  tip: "Photograph one apple leaf close up, filling most of the frame, on a plain background or on the tree.",
  retry: "Try another photo",
  tips: "Photo tips",
  anyway: "Analyse anyway",
  safeguard:
    "The check is a safeguard, not a guarantee: it mostly turns away photos that are not leaves, and it lets some leaves of other trees (pear, walnut, grape) through.",
  uncheckedLabel: "Unchecked input: this photo did not look like an apple leaf, so treat the result with extra caution.",
  tipList: [
    "One apple leaf per photo, filling most of the frame.",
    "A plain background (paper, a table) or the leaf on the tree, in daylight.",
    "The upper side of the leaf, in focus, without flash glare.",
    "The detector looks for apple scab lesions only; it cannot see other diseases.",
  ],
} as const;

/** The API's own warnings about the input check, which the leaf scan shows as a notice or label instead. */
export function isInputCheckWarning(warning: string): boolean {
  return warning.includes("skip_input_check");
}
