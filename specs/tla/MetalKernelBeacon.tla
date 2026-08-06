--------------------------- MODULE MetalKernelBeacon ---------------------------
(***************************************************************************)
(* Model of the metal-kernel-beacon parent/child protocol.                *)
(*                                                                         *)
(* skinny runs GPU work in a CHILD process (the metal-dispatch-hygiene     *)
(* timeout pattern). The child stamps a kernel id into a shared beacon     *)
(* cell BEFORE it submits each Metal dispatch, then flushes. The PARENT    *)
(* polls the cell on a wall clock. On timeout the parent SIGTERMs the      *)
(* child and reads the last stamped cell to report which kernel hung.      *)
(*                                                                         *)
(* macOS cannot cancel a committed command buffer, so an infinite kernel   *)
(* wedges the child in wait_for_idle forever. The pre-dispatch stamp is    *)
(* the load-bearing record: it lands before the dispatch that may hang.    *)
(*                                                                         *)
(* This module abstracts the shared beacon cell as THE value the parent    *)
(* reads (the mmap cell written by the host stamp). The separate UMA GPU   *)
(* beacon buffer only ever confirms the same kernel the child is already   *)
(* dispatching, so it cannot change what the parent reports and is not     *)
(* modeled.                                                                *)
(*                                                                         *)
(* The stamp is a seqlock: the child sets the cell TORN while it writes    *)
(* the payload, then clears TORN. A reader that sees TORN (seq != seq_check *)
(* in the real layout) treats the cell as "no valid beacon". This models   *)
(* the torn-read guard.                                                    *)
(*                                                                         *)
(* BOUNDS / ASSUMPTIONS (small, bounded model):                            *)
(*  - KernelSeq is the fixed short list of kernels the child dispatches in  *)
(*    order (design-gate's .cfg sets it, e.g. <<1, 2>> — 2 kernels).        *)
(*  - Kernel ids are the values in KernelSeq; NONE = 0 is not a kernel.     *)
(*  - The parent's wall clock is LONGER than any healthy kernel, so the     *)
(*    parent times out ONLY on a genuine hang. This is modeled by gating    *)
(*    TimedOut on childPhase = "hung".                                      *)
(***************************************************************************)

EXTENDS Naturals, Sequences

CONSTANTS KernelSeq          \* fixed sequence of kernel ids the child runs

\* TLC's .cfg grammar cannot take a sequence literal on a CONSTANT. The cfg
\* routes KernelSeq through `CONSTANT KernelSeq <- KernelSeqDef` instead.
KernelSeqDef == <<1, 2>>     \* 2 kernels, ids 1 and 2 (the bounded model)

NONE == 0                    \* "no kernel" sentinel (KERNEL_ID_NONE)

Kernels == { KernelSeq[i] : i \in DOMAIN KernelSeq }

ASSUME NONE \notin Kernels   \* kernel ids are >= 1; 0 means "no kernel"

VARIABLES
    idx,          \* next kernel index the child will stamp (1..Len+1)
    childPhase,   \* "idle" | "stamping" | "dispatching" | "hung"
    cellKid,      \* kernel id currently in the shared beacon cell
    cellTorn,     \* TRUE while the child is mid-stamp (seqlock open)
    parentPhase,  \* "polling" | "timed_out" | "sigterm_sent" | "read_beacon"
    reported,     \* kernel id the parent reported (NONE until it reads)
    reached       \* ghost: kernel ids the child has actually reached (stamped)

vars == <<idx, childPhase, cellKid, cellTorn, parentPhase, reported, reached>>

SeqLen == Len(KernelSeq)

(* What a reader observes right now: a torn cell reads as NONE.           *)
ReadValue == IF cellTorn THEN NONE ELSE cellKid

--------------------------------------------------------------------------------

Init ==
    /\ idx = 1
    /\ childPhase = "idle"
    /\ cellKid = NONE
    /\ cellTorn = FALSE
    /\ parentPhase = "polling"
    /\ reported = NONE
    /\ reached = {}

(* Child opens the seqlock and writes the kernel id (cell now TORN).       *)
(* The kernel is "reached" the instant its id lands in the cell, so the    *)
(* cell can never name a future kernel.                                    *)
StartStamp ==
    /\ childPhase = "idle"
    /\ idx =< SeqLen
    /\ cellKid' = KernelSeq[idx]
    /\ cellTorn' = TRUE
    /\ reached' = reached \cup { KernelSeq[idx] }
    /\ childPhase' = "stamping"
    /\ UNCHANGED <<idx, parentPhase, reported>>

(* Child closes the seqlock (flush done). This is "wrote_beacon(k)".       *)
FinishStamp ==
    /\ childPhase = "stamping"
    /\ cellTorn' = FALSE
    /\ childPhase' = "dispatching"
    /\ UNCHANGED <<idx, cellKid, parentPhase, reported, reached>>

(* Healthy kernel: wait_for_idle returns; move to the next kernel.         *)
Complete ==
    /\ childPhase = "dispatching"
    /\ idx' = idx + 1
    /\ childPhase' = "idle"
    /\ UNCHANGED <<cellKid, cellTorn, parentPhase, reported, reached>>

(* Infinite kernel: wait_for_idle never returns. childPhase stays "hung".  *)
Hang ==
    /\ childPhase = "dispatching"
    /\ childPhase' = "hung"
    /\ UNCHANGED <<idx, cellKid, cellTorn, parentPhase, reported, reached>>

(* Parent's wall clock expires. It fires only on a genuine hang (see the   *)
(* wall-clock assumption above).                                           *)
TimedOut ==
    /\ parentPhase = "polling"
    /\ childPhase = "hung"
    /\ parentPhase' = "timed_out"
    /\ UNCHANGED <<idx, childPhase, cellKid, cellTorn, reported, reached>>

(* Parent SIGTERMs the child (the chained handler runs destroy()).         *)
SigTerm ==
    /\ parentPhase = "timed_out"
    /\ parentPhase' = "sigterm_sent"
    /\ UNCHANGED <<idx, childPhase, cellKid, cellTorn, reported, reached>>

(* Parent reads the last valid beacon cell. A torn cell reads as NONE.     *)
ReadBeacon ==
    /\ parentPhase = "sigterm_sent"
    /\ reported' = ReadValue
    /\ parentPhase' = "read_beacon"
    /\ UNCHANGED <<idx, childPhase, cellKid, cellTorn, reached>>

Next ==
    \/ StartStamp
    \/ FinishStamp
    \/ Complete
    \/ Hang
    \/ TimedOut
    \/ SigTerm
    \/ ReadBeacon

Fairness ==
    /\ WF_vars(TimedOut)
    /\ WF_vars(SigTerm)
    /\ WF_vars(ReadBeacon)

Spec == Init /\ [][Next]_vars /\ Fairness

--------------------------------------------------------------------------------
(* Type invariant.                                                         *)
TypeOK ==
    /\ idx \in 1..(SeqLen + 1)
    /\ childPhase \in {"idle", "stamping", "dispatching", "hung"}
    /\ cellKid \in {NONE} \cup Kernels
    /\ cellTorn \in BOOLEAN
    /\ parentPhase \in {"polling", "timed_out", "sigterm_sent", "read_beacon"}
    /\ reported \in {NONE} \cup Kernels
    /\ reached \subseteq Kernels

(* SAFETY: the parent never reports a kernel the child never reached.      *)
NoPhantomReport == (reported = NONE) \/ (reported \in reached)

(* SAFETY: no torn-read or stale/future read yields a kernel id. A torn    *)
(* cell reads as NONE; a committed cell names a reached kernel.            *)
NoTornOrStaleRead == ReadValue \in ({NONE} \cup reached)

(* SAFETY: once the parent has read after a real hang, its report names a  *)
(* kernel the child actually reached (never NONE, never a phantom).        *)
ReportIsRealKernel == (parentPhase = "read_beacon") => (reported \in reached)

(* LIVENESS: a child stuck in a hang is eventually reported.               *)
HungIsReported == (childPhase = "hung") ~> (parentPhase = "read_beacon")

================================================================================
