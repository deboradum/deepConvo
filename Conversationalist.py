
class Conversationalist:
    def __init__(self, llm, global_history, role, system_prompt, verbose):
        self.llm = llm
        self.global_history = global_history
        # Role is the name/ type of the roleplay character this Conversationalist
        # plays.
        self.role = role
        self.history = [{"role": "system", "content": system_prompt}]
        self.verbose = verbose

    def talk(self):
        resp = self.llm.run(self.history)
        self.global_history.append({"role": self.role, "content": resp})
        self.history.append({"role": "assistant", "content": resp})

        if self.verbose:
            print(f"{self.role}: {resp}\n")

        return resp

    def add_partner_response(self, response):
        self.history.append({"role": "user", "content": response})
