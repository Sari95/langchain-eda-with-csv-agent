#mini_eval.py
from main import ask_agent

tests = [
    ("Welche Spalten haben Missing Values?", ["age", "embarked", "deck", "embark_town"]),
    ("Zeig mir die ersten 3 Spalten mit ihren Datentypen.", ["survived", "pclass", "sex"]),
    ("Gib mir eine statistische Zusammenfassung der Spalte 'age'.", ["mean", "min", "max"]),
]

def passed(q, out, must_include):
    text = out.lower()
    return all(any(tok in text for tok in (m.lower(), str(m).lower())) for m in must_include)

if __name__ == "__main__":
    ok = 0
    for q, must in tests:
        out = ask_agent(q)
        result = passed(q, out, must)
        print(f"[{'OK' if result else 'FAIL'}] {q}\n{out}\n")
        ok += int(result)
    print(f"Passed {ok}/{len(tests)}")

