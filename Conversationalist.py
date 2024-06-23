import json

# Writes dict_to_write to a jsonl file.
def write_file(filepath, dict_to_write):
    with open(filepath, "a+") as f:
        json_line = json.dumps(dict_to_write)
        f.write(json_line + "\n")


class Conversationalist:
    def __init__(self, llm, global_history, role, system_prompt, filepath, verbose=False):
        self.llm = llm
        self.global_history = global_history
        # Role is the name/ type of the roleplay character this Conversationalist
        # plays.
        self.role = role
        self.history = [{"role": "system", "content": system_prompt}]
        self.verbose = verbose
        self.filepath = filepath

    def talk(self):
        resp = self.llm.run(self.history)
        global_history_entry = {"role": self.role, "content": resp}
        self.global_history.append(global_history_entry)
        write_file(self.filepath, global_history_entry)

        self.history.append({"role": "assistant", "content": resp})

        if self.verbose:
            print(f"{self.role}: {resp}\n")

        return resp

    def add_partner_response(self, response):
        self.history.append({"role": "user", "content": response})
