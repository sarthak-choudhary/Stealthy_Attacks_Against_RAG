MULTIPLE_BINARY_PROMPT = "You are a helpful assistant. Below is a user's query and relevant contexts. \
Answer the query in one word only: 'YES', 'NO', or 'I don’t know'. \
Provide 'I don’t know' only if the context does not have enough information to answer. \
\n\nContexts: [context] \n\nQuery: [question] \n\nThe answer to the query is "

MULTIPLE_CHOICE_PROMPT = "You are a helpful assistant. Below is a multiple choice question and relevant contexts. \
    Answer the question using the choices. Only give the correct choice as the response without anything else.\
    \n\nContexts: [context] \n\nQuestion: [question] \n\nChoices: [choices] \n\nThe answer to the questions is "

MULTIPLE_PROMPT = "You are a helpful assistant. Below is a user's query and relevant contexts. \
Answer the query using as less words as possible. Try to use as less words as possible. \
\n\nContexts: [context] \n\nQuery: [question] \n\nThe answer to the query is "

QA_MC_PROMPT = (
    "Conext information is below.\n"
    "-------------------------\n"
    "[context]\n"
    "-------------------------\n"
    "Given the context information and not prior knowledge, "
    "try to find the best candidate answer to the query. \n"
    "Query: [question]\n"
    "Candidates: \nA. [A] \nB. [B] \nC. [C] \nD. [D] \nE. No information found \n"
    "Output an answer from A, B, C or D only when there is clear evidence found in the context information. "
    "Otherwise, output 'E. No information found' as the answer. Only output the value of the correct option from Candidates. "
    "Do not give any explaination with the answer. \n"
    "Answer: \n"
)

QA_PROMPT_TMPL =   (
    "Context information is below.\n"
    "---------------------\n"
    "[context]\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: [question]\n"
    "Answer: "
)

QA_PROMPT_WITH_EXAMPLES_TMPL =   (

    "Context information is below.\n"
    "---------------------\n"
    '''NASA's Artemis Program Advances
    In 2022, NASA made significant progress in the Artemis program, aimed at returning humans to the Moon and establishing a sustainable presence by the end of the decade... \n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What is the primary goal of NASA's Artemis program?\n"
    "Answer: Return humans to the Moon\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    '''2022 US Women's Open Highlights
    The 2022 US Women’s Open was concluded in June at Pine Needles Lodge & Golf Club in North Carolina. Minjee Lee emerged victorious capturing ... \n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: Which golfer won the 2022 US Women’s Open?\n"
    "Answer: Minjee Lee\n\n\n"
    
    "Context information is below.\n"
    "---------------------\n"
    '''Microsoft acquires gaming company
    Microsoft has completed the acquisition of the gaming company Activision Blizzard. This move is expected to enhance Microsoft's gaming portfolio and significantly boost its market share in the gaming industry...\n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new video game titles are being released by Microsoft this year?\n"
    "Answer: I don't know\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    '''Apple launches iPhone 14 with satellite connectivity
    Apple has officially launched the iPhone 14, which includes a groundbreaking satellite connectivity feature for emergency situations. This feature is designed to ensure safety in remote areas without cellular service...\n'''
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: What new feature does the iPhone 14 have?\n"
    "Answer: Satellite connectivity\n\n\n"

    "Context information is below.\n"
    "---------------------\n"
    "[context]\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don\'t know\".\n" 
    "Query: [question]\n"
    "Answer: "
)
prompt_types = {
    "realtimeqa": QA_MC_PROMPT,
    "realtimeqa_nochoices":  QA_PROMPT_WITH_EXAMPLES_TMPL,
    "open_nq": QA_PROMPT_WITH_EXAMPLES_TMPL,
    "wiki_open_nq": QA_PROMPT_WITH_EXAMPLES_TMPL,
    "wiki_hotpotqa": QA_PROMPT_WITH_EXAMPLES_TMPL,
}

def wrap_prompt(question, context, choices, dataset, prompt_id=1) -> str:
    if type(context) == list:
        context_str = '\n\n'.join(f"[{i+1}] {s}" for i, s in enumerate(context))
    else:
        context_str = context

    prompt = prompt_types[dataset]
    prompt = prompt.replace("[question]", question).replace("[context]", context_str)
    if dataset == "realtimeqa":
        input_prompt = prompt.replace("[A]", choices[0]).replace("[B]", choices[1]).replace("[C]", choices[2]).replace("[D]", choices[3])
    else:
        input_prompt = prompt       
    return input_prompt

