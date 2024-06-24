from LlmClient import MlxLlama
from Conversationalist import Conversationalist
from Examples import (
    drugs_example,
    flatearth_example,
    cartman_kyle_business_example,
    randy_marsh_example,
)

conversation = []


def converse(c1, c2, convo_duration=10):
    for _ in range(convo_duration):
        c1_turn = c1.talk()
        c2.add_partner_response(c1_turn)

        c2_turn = c2.talk()
        c1.add_partner_response(c2_turn)


if __name__ == "__main__":
    # Get roles and prompts.
    role1, role1_sysprompt, role2, role2_sysprompt, convo_file = (
        cartman_kyle_business_example()
    )

    llm = MlxLlama(temp=0.0)
    c1 = Conversationalist(llm, conversation, role1, role1_sysprompt, convo_file, True)
    c2 = Conversationalist(llm, conversation, role2, role2_sysprompt, convo_file, True)

    converse(c1, c2, 15)
