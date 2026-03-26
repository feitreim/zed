; Python text concealments — visual-only substitutions.
; Each @conceal capture with a conceal.replacement property
; will be displayed as the replacement text in the editor.

("lambda" @conceal (#set! conceal.replacement "λ"))
("!=" @conceal (#set! conceal.replacement "≠"))
("==" @conceal (#set! conceal.replacement "≡"))
("->" @conceal (#set! conceal.replacement "→"))
("and" @conceal (#set! conceal.replacement "∧"))
("or" @conceal (#set! conceal.replacement "∨"))
("not" @conceal (#set! conceal.replacement "¬"))
("in" @conceal (#set! conceal.replacement "∈"))
("not in" @conceal (#set! conceal.replacement "∉"))
