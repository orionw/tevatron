from datasets import load_dataset
import random
from collections import defaultdict
import tqdm
import json

# llaam3= """You are an expert at writing precise detailed instructions for language models and paid millions of dollars to be a prompt engineer for OpenAI. Answer succinctly and follow all instructions given. I have the following queries and 5 documents which have previously been annotated as relevant to this query: Query: how long can you collect unemployment in california Doc A: To request benefit payments, you must certify for benefits by submitting a certification online, by phone, or by mail.\nNote: You must serve a one-week unpaid waiting period on your claim before you are paid UI benefits. The waiting period can only be served if you certify for benefits and meet all eligibility requirements for that week. Your first certification will usually include the one-week unpaid waiting period and one week of payment if you meet eligibility requirements for both weeks. Certify for benefits every two weeks to continue receiving benefit payments. Doc B: In most cases, you can receive unemployment benefits in California for up to 26 weeks. During periods of high unemployment or during a crisis such as the COVID-19 pandemic, additional weeks of benefits might be available. Doc C: If you’re self-employed or an independent contractor, you are not automatically eligible because your employer — you — didn’t pay the taxes that fund the UI program. You’ll be eligible only if you worked as an employee within the previous 18 months for a company that paid into the system on your behalf. Doc D: The amount of unemployment insurance you’ll get per week depends on your wages during the recent 12-month period. It could range from $40 to $450. This is about half of your weekly earnings. EDD provides a UI calculator to help you determine how much you’re qualified to receive.\n\nThe benefits will last for 26 weeks, given that you don’t get a new job before that period that lets you earn more than what you’re getting with unemployment insurance.  Doc E: The unemployment insurance program in California is run by the state’s Employment Development Department, or EDD. This agency allows people to file unemployment claims online or by phone, fax or mail. Typically, California unemployment benefits are available for a maximum of 26 weeks. ## Your task I need you to come up with an instruction that can be appended onto the end of these queries that will make two documents relevant but three document non-relevant. This additional instruction should provide a test for strong frontier language models to determine if they can follow instructions. For that reason, don't make the instruction so obvious to A and B, instead make them have to reason about it. For this example, please generate the instruction to be very long, approximately 2 paragraphs. Your instruction is allowed to contain background information. Your instruction may also be written from the POV of a persona, where you describe why you are searching for this information. Output the response in JSON form only with no other text, with the keys, "instruction" (str), "relevant_docs" (a list) and "non-relevant_docs" (a list). ## Your output (JSON only):"""

random.seed(123456789)

print(f"Loading dataset")
dataset = load_dataset("Tevatron/msmarco-passage-aug", trust_remote_code=True)
# breakpoint()
"""
need to take in the original, match on query_id and and then append to the query the instruction on various amounts
Dataset({
    features: ['query_id' (str), 'query' (str), 'positive_passages' (List[Dict["title", "text", "docid" all strings]]), 'negative_passages'],
    num_rows: 491007
})
"""
# get all queries
queries = {}
# query2posneg = {}
for idx, inst in tqdm.tqdm(enumerate(dataset["train"])):
    query = inst["query"]
    queries[str(idx)] = query
    # get pos and negative passages
    # pos = [item["docid"] for item in inst["positive_passages"]]
    # neg = [item["docid"] for item in inst["negative_passages"]]
    # query2posneg[str(idx)] = {"pos": pos, "neg": neg}

# save to file
# with open("tevatron_queries.json", "w") as fOut:
#     json.dump(queries, fOut, indent=4)

# do queries as a tsv
with open("tevatron_queries.tsv", "w") as fOut:
    for key, value in queries.items():
        fOut.write(f"{key}\t{value}\n")

# with open("tevatron_query2posneg.json", "w") as fOut:
#     json.dump(query2posneg, fOut, indent=4)








# full_example = """I have the following queries and 5 documents which have previously been annotated as relevant to this query:

# EXAMPLES

# ## Your task
# I need you to come up with an instruction that can be appended onto the end of these queries that will make Docs A and B relevant but docs C, D, and E non-relevant. This additional instruction should provide a test for strong frontier language models to determine if they can follow instructions. For that reason, don't make the instruction so obvious to A and B, instead make them have to reason about it.

# ## Your instruction:"""

# template = """Query {LETTER}: {QUERY}\nDoc {LETTER}: {DOC}\n\n"""

# queries_ids = dataset["train"]["query_id"]

# # find groups of them that have 5 docs
# query_ids_map = defaultdict(list)
# for i, inst in enumerate(queries_ids):
#     query_ids_map[inst].append(i)

# # keep those which have the highest number of docs
# query_ids_to_keep = []
# for key, value in query_ids_map.items():
#     if len(value) >= 5:
#         query_ids_to_keep.append(value)
# breakpoint()


# train_instances = []
# task_i = -1
# while len(train_instances) < 2:
#     task_i += 1
#     inst = dataset["train"][task_i]
#     breakpoint()
#     query = inst["query"]
#     examples = ""
#     pos_passages = inst["positive_passages"]
#     if len(pos_passages) < 4:
#         print(len(pos_passages))
#         continue
#     positive_passages = random.sample(pos_passages, k=4)
#     for positive_passage in positive_passages:
#         full_doc_text = positive_passage["title"] + " " + positive_passage["text"]
#         examples += template.format(LETTER=chr(task_i+65), QUERY=query, DOC=positive_passage)

#     full_example.replace("EXAMPLES", examples)
#     train_instances.append(full_example)
#     print(full_example, "\n\n")


# breakpoint()
# print("Done!")