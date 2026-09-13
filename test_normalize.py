import time
from normalize import normalize_for_tts

tests = [
    ("Jade calc", "Sent to **jade** — crunching `54 × 54`. Let me know when it is done!"),
    
    ("Jade done", "**Jade** is done\n\n> **54 × 54 = 2916**\n\nCompleted in ~2 seconds, 1 turn. Also benefited from cache hits this time — cost was only ~$0.016 vs golds ~$0.040."),
    
    ("Hungarian", "Úgy tűnik, hogy véletlenül küldted el ezt — talán egy másik alkalmazásból?\n\nHa **képernyőmegosztást** és **üzenetküldést** szeretnél az Iris OS-be építeni, szívesen segítek. Pontosítanád, hogy mire gondoltál?"),
    
    ("Bye", "Bye Paul! See you next time!"),
    
    ("Lime exploring", "Limes on it — exploring the eye-on-rainbow project now. I will let you know what it finds."),
    
    ("Suite found", "Heres what lime found:\n\n**`suite.new.json`** is a migration-in-progress — its the UI-based, single page-load version of `suite.old`. Currently has **3 of 9** workflows:\n\nAlready in:\n- `flex.workspace-creation`\n- `flex.reservation-approval-flow`\n- `flex.process-crud`\n\nStill missing:\n- `flex.machine-crud`\n- `flex.recipe-crud`\n- `flex.inventory-crud`\n\nWhat do you want to do — add the missing workflows, or something else?"),
    
    ("STT garble", "That doesnt quite make sense in context — are you using speech-to-text? Might be garbling things. What did you mean to say?"),
    
    ("Full picture", "Heres the full picture:\n\n**`suite.new`** (3 workflows):\n- `flex.workspace-creation` ← only in new (UI-based login + create)\n- `flex.reservation-approval-flow` ← shared\n- `flex.process-crud` ← shared\n\n**`suite.old`** (9 workflows) — 6 still not migrated to new:\n- `flex.workspace-setup` (replaced by `workspace-creation`)\n- `flex.machine-crud`\n- `flex.recipe-crud`\n\nSo `suite.new` is about a third done. What do you want to work on?"),
]

print("=" * 70)
print("REAL IRIS RESPONSES TEST")
print("=" * 70)

total_ms = 0
for name, text in tests:
    print("\n>>> [%s]" % name)
    start = time.time()
    result = normalize_for_tts(text)
    ms = (time.time() - start) * 1000
    total_ms += ms
    print("TIME: %dms" % ms)
    out = result.replace("\n", " ")
    print("OUT:", out[:150])
    if len(out) > 150:
        print("     ", out[150:300])
    print()

print("=" * 70)
print("AVG: %dms" % (total_ms / len(tests)))
