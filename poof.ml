let bot = (fun _ -> assert false)

let fix proto super =
  let rec f () = proto (fun i -> (f ()) i) super in
  f ()

let mix c p = fun f b -> c f (p f b)

let mux q r = fun f b -> fix (mix q r) bot

let () = 
  mux (a) (mix b c)
  = mux a (fun f b -> c b (p b c))

(* the current interpretation *)
let i = ref (fun self super _ -> assert false)

(* run f with intp as a handler *)
let handler intp f = 
  let old = !i in
  i := mix intp old; 
  f ();
  i := old

(* run f with intp as a runner *)
let runner intp f = 
  let old = !i in 
  i := mux intp old; 
  f ();
  i := old

(* perform an operation *)
let perform args = (fix !i bot) args

type dyn = Int of int | List of dyn list
type op = Double of dyn | Triple of dyn | Sextuple of dyn

let rec print_dyn = function
  | Int v -> print_int v
  | List l -> print_string "[";
              List.iter (fun x -> print_dyn x; print_string "; ") l;
              print_string "]"

let () = 
  let multiply_in_length self super = function
    | Double v -> List [v; v]
    | Triple v -> List [v; v; v]
    | msg -> super msg
  in

  let multiply_in_value self super = function
    | Double (Int v) -> Int (v * 2)
    | Triple (Int v) -> Int (v * 3)
    | msg -> super msg
  in

  let sextuple_as_double_triple self super = function 
    | Sextuple v -> self (Double (self (Triple v)))
    | msg -> super msg
  in

  handler multiply_in_length (fun () ->
      handler sextuple_as_double_triple (fun () ->
          handler multiply_in_value (fun () ->
              assert ((perform (Double (Int 2))) = Int 4);
              assert ((perform (Triple (Int 3))) = Int 9);
              assert ((perform (Sextuple (Int 6))) = Int 36)
            )
        );

      runner sextuple_as_double_triple (fun () ->
          handler multiply_in_value (fun () ->
              assert ((perform (Double (Int 2))) = Int 4);
              assert ((perform (Triple (Int 3))) = Int 9);
              assert ((perform (Sextuple (Int 6))) = List [List [Int 6; Int 6; Int 6]; List [Int 6; Int 6; Int 6]])
            )
        )
    )
