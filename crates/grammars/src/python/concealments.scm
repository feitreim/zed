; Python text concealments — visual-only substitutions.
; Each @conceal capture with a conceal.replacement property
; will be displayed as the replacement text in the editor.

; Keywords
("lambda" @conceal (#set! conceal.replacement "λ"))
("return" @conceal (#set! conceal.replacement "⏎"))

; Comparison / equality — scoped to comparison_operator to avoid
; matching inside unrelated tokens.
(comparison_operator "!=" @conceal (#set! conceal.replacement "≠"))
(comparison_operator "==" @conceal (#set! conceal.replacement "≡"))
(comparison_operator ">=" @conceal (#set! conceal.replacement "≥"))
(comparison_operator "<=" @conceal (#set! conceal.replacement "≤"))

; Type annotations
("->" @conceal (#set! conceal.replacement "→"))

; Boolean operators — scoped so "and"/"or" only match inside
; boolean_operator, not inside keywords like "for".
(boolean_operator "and" @conceal (#set! conceal.replacement "∧"))
(boolean_operator "or" @conceal (#set! conceal.replacement "∨"))
(not_operator "not" @conceal (#set! conceal.replacement "¬"))

; Membership — scoped to comparison_operator so "in" in for-loops
; is not concealed.
(comparison_operator "in" @conceal (#set! conceal.replacement "∈"))
(comparison_operator "not in" @conceal (#set! conceal.replacement "∉"))

; Arithmetic
(binary_operator "**" @conceal (#set! conceal.replacement "^"))
(binary_operator "//" @conceal (#set! conceal.replacement "÷"))
