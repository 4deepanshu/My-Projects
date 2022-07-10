(define (domain robot)
(:requirements :strips)
(:predicates
    (robot ?x)
    (location ?x)
    (key ?x)
    (connected ?l1 ?l2)

    (at ?x ?y)
    (holding ?r ?k)
    (free ?l)
    (locked ?l ?k)

    (fuel ?f)
    (fuel-predecessor ?f1 ?f2)
    (fuel-level ?r ?f)
)

(:action move-robot
 :parameters (?r ?f ?t ?f1 ?f2)
 :precondition (and (robot ?r) (location ?f) (location ?t)
                    (fuel ?f1) (fuel ?f2)
                    (at ?r ?f) (not (at ?r ?t)) (free ?t) (fuel-level ?r ?f2)
                    (fuel-predecessor ?f1 ?f2)  (connected ?f ?t) (not (free ?f))
               )
 :effect (and (at ?r ?t) (not (at ?r ?f)) (free ?f) (not (free ?t)) (fuel-level ?r ?f1) (not (fuel-level ?r ?f2)) 
         )
)

(:action take-key
 :parameters (?r ?k ?l)
 :precondition (and (robot ?r) (key ?k) (location ?l)
                (at ?r ?l) (at ?k ?l)
               )
 :effect (and (holding ?r ?k)
         )
)

(:action drop-key
 :parameters (?r ?k ?l)
 :precondition (and (robot ?r) (key ?k) (location ?l)
                (holding ?r ?k) (at ?r ?l)
               )
 :effect (and (not (holding ?r ?k)) (at ?k ?l)
         )
)

(:action unlock
 :parameters (?r ?k ?f ?t)
 :precondition (and (robot ?r) (key ?k) (location ?f) (location ?t)
                (connected ?f ?t) (at ?r ?f) (not (at ?r ?t)) (holding ?r ?k) (locked ?t ?k) (not (free ?t))
               )
 :effect (and (at ?r ?f) (not (at ?r ?t)) (not (locked ?t ?k)) (free ?t)
         )
)

)