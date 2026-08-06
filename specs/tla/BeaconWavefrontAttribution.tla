------------------------ MODULE BeaconWavefrontAttribution ------------------------
(***************************************************************************)
(* Model of the beacon-wavefront-attribution fix.                          *)
(*                                                                         *)
(* The metal-kernel-beacon child stamps a kernel id into a shared mmap     *)
(* cell BEFORE it encodes each Metal dispatch. On the WAVEFRONT path the   *)
(* MetalFrameEncoder batches many kernels into ONE command buffer and      *)
(* submits ONCE, so each stamp overwrites the cell before the single       *)
(* submit. When the GPU wedges on a kernel k mid-batch, the cell holds the *)
(* LAST-ENCODED kernel of the batch, not k. The parent SIGTERMs the child  *)
(* and reads the cell, so it reports the WRONG kernel.                     *)
(*                                                                         *)
(* The fix (design B): under SKINNY_METAL_TRACE the encoder degrades to    *)
(* per-kernel submit. It stamps, submits ONE kernel, and drains before it  *)
(* encodes the next, so at most one kernel is in flight per command buffer *)
(* and the cell always names the in-flight kernel.                         *)
(*                                                                         *)
(* This module models BOTH modes through the boolean CONSTANT Trace:       *)
(*   Trace = FALSE  -> batched mode (the HAZARD; violates NoMisattribution) *)
(*   Trace = TRUE   -> per-kernel submit (the FIX; satisfies it)            *)
(* The design gate's .cfg runs the module under each value to demonstrate  *)
(* the fix.                                                                 *)
(*                                                                         *)
(* Ghost variables:                                                        *)
(*   inflight  = the kernel actually running / hung on the GPU.            *)
(*   reported  = the kernel the parent reads from the mmap after SIGTERM.  *)
(*                                                                         *)
(* SAFETY NoMisattribution: after the parent reads, reported = inflight.   *)
(*                                                                         *)
(* BOUNDS / ASSUMPTIONS (small, bounded model):                            *)
(*  - KernelSeq is a fixed short list of kernel ids the child encodes in    *)
(*    order (KernelSeqDef, e.g. <<1, 2, 3>> — a batch of 3).                *)
(*  - Kernel ids are the values in KernelSeq; NONE = 0 is not a kernel.     *)
(*  - The parent's wall clock is LONGER than any healthy kernel, so it      *)
(*    times out ONLY on a genuine hang (TimedOut is gated on phase="hung"). *)
(*  - The batched mode picks ANY kernel of the in-flight batch to hang,     *)
(*    modeling that the wedge can land on any kernel of the command buffer. *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    KernelSeq,   \* fixed sequence of kernel ids the child encodes, in order
    Trace        \* TRUE = per-kernel submit (fix); FALSE = batched (hazard)

\* TLC's .cfg grammar cannot take a sequence literal on a CONSTANT, so the cfg
\* routes KernelSeq through `CONSTANT KernelSeq <- KernelSeqDef`.
KernelSeqDef == <<1, 2, 3>>   \* 3 kernels, ids 1..3 (the bounded batch)

NONE == 0                     \* "no kernel" sentinel (KERNEL_ID_NONE)

Kernels == { KernelSeq[i] : i \in DOMAIN KernelSeq }

ASSUME NONE \notin Kernels    \* kernel ids are >= 1; 0 means "no kernel"
ASSUME Trace \in BOOLEAN

VARIABLES
    idx,          \* next kernel index the child will encode (1..Len+1)
    batch,        \* kernel ids encoded into the current OPEN command buffer
    flying,       \* kernel ids of the command buffer currently in flight
    cellKid,      \* kernel id currently in the shared mmap cell
    cellTorn,     \* TRUE while the child is mid-stamp (seqlock open)
    phase,        \* "encoding" | "hung"
    inflight,     \* ghost: kernel actually running / hung (NONE = none)
    parentPhase,  \* "polling" | "timed_out" | "sigterm_sent" | "read_beacon"
    reported,     \* kernel id the parent reported (NONE until it reads)
    reached       \* ghost: kernel ids the child has actually reached (stamped)

vars == <<idx, batch, flying, cellKid, cellTorn, phase, inflight,
          parentPhase, reported, reached>>

SeqLen == Len(KernelSeq)

(* What a reader observes right now: a torn cell reads as NONE.            *)
ReadValue == IF cellTorn THEN NONE ELSE cellKid

--------------------------------------------------------------------------------

Init ==
    /\ idx = 1
    /\ batch = {}
    /\ flying = {}
    /\ cellKid = NONE
    /\ cellTorn = FALSE
    /\ phase = "encoding"
    /\ inflight = NONE
    /\ parentPhase = "polling"
    /\ reported = NONE
    /\ reached = {}

(* Child MAY encode the next kernel. In per-kernel-submit mode (Trace) it   *)
(* cannot start a second kernel while a batch is open or a command buffer   *)
(* is in flight, so at most one kernel is ever in flight. In batched mode   *)
(* it may accumulate more kernels into the open command buffer.             *)
CanEncode ==
    /\ idx =< SeqLen
    /\ flying = {}
    /\ ~cellTorn
    /\ (Trace => batch = {})

(* Child opens the seqlock and stamps the kernel id (cell now TORN). The    *)
(* kernel is "reached" the instant its id lands in the cell.                *)
EncodeStart ==
    /\ phase = "encoding"
    /\ CanEncode
    /\ cellTorn' = TRUE
    /\ cellKid' = KernelSeq[idx]
    /\ reached' = reached \cup { KernelSeq[idx] }
    /\ batch' = batch \cup { KernelSeq[idx] }
    /\ idx' = idx + 1
    /\ UNCHANGED <<flying, phase, inflight, parentPhase, reported>>

(* Child closes the seqlock (flush done). The stamp is now committed.       *)
EncodeFinish ==
    /\ cellTorn = TRUE
    /\ cellTorn' = FALSE
    /\ UNCHANGED <<idx, batch, flying, cellKid, phase, inflight,
                   parentPhase, reported, reached>>

(* Submit the open command buffer: the whole batch becomes in flight. In    *)
(* per-kernel mode the batch is a singleton, so exactly one kernel flies.   *)
Submit ==
    /\ phase = "encoding"
    /\ ~cellTorn
    /\ batch # {}
    /\ flying = {}
    /\ flying' = batch
    /\ batch' = {}
    /\ UNCHANGED <<idx, cellKid, cellTorn, phase, inflight,
                   parentPhase, reported, reached>>

(* Healthy: the whole in-flight command buffer finishes; move on.           *)
Complete ==
    /\ phase = "encoding"
    /\ flying # {}
    /\ flying' = {}
    /\ UNCHANGED <<idx, batch, cellKid, cellTorn, phase, inflight,
                   parentPhase, reported, reached>>

(* Hang: some kernel of the in-flight command buffer wedges forever. The    *)
(* wedge can land on ANY kernel of the batch. In per-kernel mode the batch  *)
(* is a singleton, so the hung kernel = the only (last-stamped) kernel.     *)
Hang ==
    /\ phase = "encoding"
    /\ flying # {}
    /\ \E k \in flying: inflight' = k
    /\ phase' = "hung"
    /\ UNCHANGED <<idx, batch, flying, cellKid, cellTorn,
                   parentPhase, reported, reached>>

(* Parent's wall clock expires. It fires only on a genuine hang.            *)
TimedOut ==
    /\ parentPhase = "polling"
    /\ phase = "hung"
    /\ parentPhase' = "timed_out"
    /\ UNCHANGED <<idx, batch, flying, cellKid, cellTorn, phase,
                   inflight, reported, reached>>

(* Parent SIGTERMs the child (the chained handler runs destroy()).          *)
SigTerm ==
    /\ parentPhase = "timed_out"
    /\ parentPhase' = "sigterm_sent"
    /\ UNCHANGED <<idx, batch, flying, cellKid, cellTorn, phase,
                   inflight, reported, reached>>

(* Parent reads the last valid beacon cell. A torn cell reads as NONE.      *)
ReadBeacon ==
    /\ parentPhase = "sigterm_sent"
    /\ reported' = ReadValue
    /\ parentPhase' = "read_beacon"
    /\ UNCHANGED <<idx, batch, flying, cellKid, cellTorn, phase,
                   inflight, reached>>

Next ==
    \/ EncodeStart
    \/ EncodeFinish
    \/ Submit
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
    /\ batch \subseteq Kernels
    /\ flying \subseteq Kernels
    /\ cellKid \in {NONE} \cup Kernels
    /\ cellTorn \in BOOLEAN
    /\ phase \in {"encoding", "hung"}
    /\ inflight \in {NONE} \cup Kernels
    /\ parentPhase \in {"polling", "timed_out", "sigterm_sent", "read_beacon"}
    /\ reported \in {NONE} \cup Kernels
    /\ reached \subseteq Kernels

(* SAFETY (the point of this change): after the parent reads, it reports    *)
(* the IN-FLIGHT (hung) kernel. Batched mode VIOLATES this — the cell holds *)
(* the last-encoded kernel, which need not be the one that hung. Per-kernel *)
(* submit SATISFIES it — one kernel flies, so the cell names it.            *)
NoMisattribution == (parentPhase = "read_beacon") => (reported = inflight)

(* SAFETY (weaker): the report is always SOME reached kernel, never a       *)
(* phantom. This HOLDS in BOTH modes, which shows the batched bug is a      *)
(* misattribution among reached kernels, not an out-of-set phantom.         *)
NoPhantomReport == (reported = NONE) \/ (reported \in reached)

(* SAFETY: while a kernel is hung the seqlock is closed, so the parent reads *)
(* a committed cell. This formalizes the seqlock quiescence assumption.     *)
WriterQuiescentWhenHung == (phase = "hung") => (cellTorn = FALSE)

(* SAFETY (fix-only): under trace, at most one kernel is in flight per       *)
(* command buffer.                                                          *)
AtMostOneInFlight == Trace => (Cardinality(flying) =< 1)

(* LIVENESS: a hung kernel is eventually reported.                          *)
HungIsReported == (phase = "hung") ~> (parentPhase = "read_beacon")

================================================================================
