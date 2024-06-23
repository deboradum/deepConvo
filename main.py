from LlmClient import MlxLlama
from Conversationalist import Conversationalist
from Examples import drugs_example

conversation = []


def converse(c1, c2, convo_duration=10):
    for _ in range(3):
        c1_turn = c1.talk()
        c2.add_partner_response(c1_turn)

        c2_turn = c2.talk()
        c1.add_partner_response(c2_turn)


if __name__ == "__main__":
    role1, role1_sysprompt, role2, role2_sysprompt = drugs_example()

    llm = MlxLlama()

    c1 = Conversationalist(llm, conversation, role1, role1_sysprompt, True)
    c2 = Conversationalist(llm, conversation, role2, role2_sysprompt, True)

    converse(c1, c2, 10)
