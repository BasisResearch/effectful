type 'a op = ..
type 'a term = App of 'a op * 'a expr list * (string * 'a expr) list
and 'a expr = Literal of 'a | Term of 'a term


type _ op += Add : int op | Mul : int op
let one = Literal 1
let two = Literal 2

let add x y = Term (App (Add, [x; y], []))
