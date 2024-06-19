!pip3 install -q -U bitsandbytes==0.42.0
!pip3 install -q -U peft==0.8.2
!pip3 install -q -U trl==0.7.10
!pip3 install -q -U accelerate==0.27.1
!pip3 install -q -U datasets==2.17.0
!pip3 install -q -U transformers==4.38.0

import os
import transformers
import torch
from google.colab import userdata
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer
from typing import Dict, Optional, List, Tuple
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

class Agent:
    def __init__(self, name, description, model_name, **kwargs):
        self.name = name
        self.description = description
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.model_name = model_name

        # Load the tokenizer

        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=os.environ['HF_TOKEN'])

        # Initialize the model with empty weights and use `init_empty_weights`

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                             quantization_config=bnb_config,
                                             device_map={"":0},
                                             token=os.environ['HF_TOKEN'])

        # Infer the device map
        # device_map = infer_auto_device_map(self.model, max_memory={0: "12GiB", "cpu": "30GiB"})

        # Dispatch the model to the appropriate devices
        # self.model = dispatch_model(self.model, device_map=device_map)

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List] = None,
        seed: int = 10,
    ) -> str:
        # Ensure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)

        # Ensure if messages is provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        # Concatenate messages into a single prompt
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Tokenize the input
        device = "cuda:0"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)

        # Generate response
        # with torch.no_grad():
            # outputs = self.model.generate(**inputs, max_length=200, do_sample=True, top_p=0.95, top_k=50, temperature=0.7)

        outputs = self.model.generate(**inputs, max_new_tokens=20)
        print("output generated")

        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def generate_reply(
        self,
        task: str,
        state: Dict,
        sender: "Agent",
    ) -> Tuple[str, Dict]:
        return (
            "This is a reply from the agent. REPLY NOT IMPLEMENTED! Terminate the whole process!",
            state,
        )


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Agent1:
    def __init__(self, name, description, model_name, **kwargs):
        self.name = name
        self.description = description
        self.system_prompt = "You're a helpful assistant."
        self.kwargs = kwargs
        self.model_name = model_name

        # Load the tokenizer
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=os.environ['HF_TOKEN'])

        # Initialize the model with empty weights and use `init_empty_weights`
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                             quantization_config=bnb_config,
                                             device_map={"": 0},
                                             token=os.environ['HF_TOKEN'])

    def llm_call(self, prompt: Optional[str] = None, messages: Optional[List] = None, seed: int = 10) -> str:
        # Ensure exactly one of prompt or messages is provided
        assert (prompt is None) != (messages is None)

        # Ensure if messages are provided, it is a list of dicts with role and content
        if messages is not None:
            assert isinstance(messages, list)
            for message in messages:
                assert isinstance(message, dict)
                assert "role" in message
                assert "content" in message

        if prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

        # Concatenate messages into a single prompt
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Tokenize the input
        device = "cuda:0"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(device)

        # Generate response
        outputs = self.model.generate(**inputs, max_new_tokens=50,
                                      num_return_sequences=1,
                                      pad_token_id=self.tokenizer.eos_token_id,
                                      )
        print("output generated")

        # Decode the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print("output decoded")
        print(response,"\n\n")
        # Extract only the assistant's response
        response_lines = response.split("\n")
        assistant_response = next((line for line in response_lines if line.startswith("assistant:")), "")
        assistant_response = assistant_response[len("assistant:"):].strip()

        print(assistant_response)
        return assistant_response

    def generate_reply(self, task: str, state: Dict, sender: "Agent") -> Tuple[str, Dict]:
        return (
            "This is a reply from the agent. REPLY NOT IMPLEMENTED! Terminate the whole process!",
            state,
        )

agent = Agent(name="TestAgent", description="A test agent", model_name="google/gemma-2b")

# Call the model with a prompt
response = agent.llm_call(prompt='''
Assume the parameters are defined. Now generate a code accordingly and enclose it between "=====" lines. Only generate the gurobi code, and don't generate any other text. Here's an example:

**input**:

{{
    "definition": "Quantity of oil i bought in month m",
    "symbol": "buy_{{i,m}}",
    "shape": ["I","M"]
}}
''')
print(response)

agent = Agent(name="TestAgent", description="A test agent", model_name="mistralai/Mistral-7B-Instruct-v0.3")

# Call the model with a prompt
response = agent.llm_call(prompt='''
Assume the parameters are defined. Now generate a code accordingly and enclose it between "=====" lines. Only generate the gurobi code, and don't generate any other text. Here's an example:

**input**:

{{
    "definition": "Quantity of oil i bought in month m",
    "symbol": "buy_{{i,m}}",
    "shape": ["I","M"]
}}
''')
print(response)

variable_definition_prompt_templates = [
    """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for defining variables of the problem.
""",
    """
Here's a variable we need you to write the code for defining:

-----
{variable}
-----

Assume the parameters are defined. Now generate a code accordingly and enclose it between "=====" lines. Only generate the code, and don't generate any other text. Here's an example:

**input**:

{{
    "definition": "Quantity of oil i bought in month m",
    "symbol": "buy_{{i,m}}",
    "shape": ["I","M"]
}}

***output***:

=====
buy = model.addVars(I, M, vtype=gp.GRB.CONTINUOUS, name="buy")
=====


- Note that the indices in symbol (what comes after _) are not a part of the variable name in code.
- Use model.addVar instead of model.addVars if the variable is a scalar.

""",
]

main_prompt_templates = {
    "constraint": [
        """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for different constraints of the problem.
""",
        """
Here's a constraint we need you to write the code for, along with the list of related variables and parameters:

-----
{context}
-----

- Assume the parameters and variables are defined, and gurobipy is imported as gp. Now generate a code accordingly and enclose it between "=====" lines.
- Only generate the code and the ===== lines, and don't generate any other text.
- If the constraint requires changing a variable's integralilty, generate the code for changing the variable's integrality rather than defining the variable again.
- If there is no code needed, just generate the comment line (using # ) enclosed in ===== lines explaining why.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Here's an example:


**input**:


{{
    "description": "in month m, it is possible to store up to storageSize_{{m}} tons of each raw oil for use later.",
    "formulation": "\(storage_{{i,m}} \leq storageSize, \quad \\forall i, m\)",
    "related_variables": [
        {{
            "symbol": "storage_{{i,m}}",
            "definition": "quantity of oil i stored in month m",
            "shape": [
                "I",
                "M"
            ]
        }}
        ],
    "related_parameters": [
        {{
            "symbol": "storageSize_{{m}}",
            "definition": "storage size available in month m",
            "shape": [
                "M"
            ]
        }}
    ]
}}

***output***:

=====
# Add storage capacity constraints
for i in range(I):
    for m in range(M):
        model.addConstr(storage[i, m] <= storageSize[m], name="storage_capacity")
=====

Take a deep breath and approach this task methodically, step by step.

""",
    ],
    "objective": [
        """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to write {solver} code for the objective function of the problem.
""",
        """
Here's the objective function we need you to write the code for, along with the list of related variables and parameters:

-----
{context}
-----

Assume the parameters and variables are defined, and gurobipy is imported as gp. Now generate a code accordingly and enclose it between "=====" lines. Only generate the code and the =====s, and don't generate any other text. Here's an example:

**input**:

{{
    "description": "Maximize the total profit from selling goods",
    "formulation": "Maximize \(Z = \sum_{{k=1}}^{{K}} \sum_{{i=1}}^{{I}} (profit_k \cdot x_{{k,i}} - storeCost \cdot s_{{k,i}})\)",
    "related_variables": [
        {{
            "symbol": "x_{{k,i}}",
            "definition": "quantity of product k produced in month i",
            "shape": [
                "K",
                "I"
            ],
            "code": "x = model.addVars(K, I, vtype=gp.GRB.CONTINUOUS, name='x')"
        }},
        {{
            "symbol": "s_{{k,i}}",
            "definition": "quantity of product k stored in month i",
            "shape": [
                "K",
                "I"
            ],
            "code": "s = model.addVars(K, I, vtype=gp.GRB.CONTINUOUS, name='s')"
        }}
    ],
    "related_parameters": [
        {{
            "symbol": "profit_{{k}}",
            "definition": "profit from selling product k",
            "shape": [
                "K"
            ]
        }},
        {{
            "symbol": "storeCost",
            "definition": "price of storing one unit of product",
            "shape": []
        }}
    ]
}}


***output***:

=====
# Set objective
m.setObjective(gp.quicksum(profit[k] * x[k, i] - storeCost * s[k, i] for k in range(K) for i in range(I)), gp.GRB.MAXIMIZE)
=====

Take a deep breath and approach this task methodically, step by step.

""",
    ],
}

debugging_prompt_templates = [
    """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for {target} of the problem.
""",
    """


When running a code snippet, an error happened. Here is the initial part of the code snippet for importing packages and defining the model:

-----
{prep_code}
-----

And here is the code for defining the related parameters and variables:

-----
{context}
-----

And the error happened when running this line:

-----
{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the import code is correct. First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",
    "reason": ?
}}

Otherwise, fix the code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": ?
}}


- Note that the fixed code should be the fixed version of the original error line, not the whole code snippet.
- Do not generate any text after the json file. All the imports and model definition are already done, and you should only generate the fixed code to be replaced with the original error line.

""",
]

debugging_refined_template_target = """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for of the problem.

When running the following code snipper, an error happened:

-----
{prep_code}

{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the code for importing packages and defining parameters and variables is correct, and the error is because of the this last part which is for modeling the {target}:

-----
{error_line}
-----

First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",
    "reason": "A string explaining why you think the problem is in the formulation"
}}

otherwise, fix the last part code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": "A sting representing the fixed {target} modeling code to be replaced with the last part code"
}}

- Note that the fixed code should be the fixed version of the last part code, not the whole code snippet. Only fix the part that is for modeling the {target}.
- Do not generate any text after the json file.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Take a deep breath and solve the problem step by step.

"""

debugging_refined_template_variable = """
You're an expert programmer in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to debug the code for of the problem.

When running the following code snipper, an error happened:

-----
{prep_code}

{error_line}
-----

and here is the error message:

-----
{error_message}
-----

We know that the code for importing packages and defining parameters and variables is correct, and the error is because of the this last part which is for modeling the {target}:

-----
{error_line}
-----

First reason about the source of the error. Then, if the code is correct and the problem is likely to be in the formulation, generate a json in this format (the reason is why you think the problem is in the formulation):

{{
    "status": "correct",
    "reason": "A string explaining why you think the problem is in the formulation"
}}

otherwise, fix the last part code and generate a json file with the following format:

{{
    "status": "fixed",
    "fixed_code": "A sting representing the fixed {target} modeling code to be replaced with the last part code"
}}

- Note that the fixed code should be the fixed version of the last part code, not the whole code snippet. Only fix the part that is for defining the {target}.
- Do not generate any text after the json file.
- Variables should become before parameters when defining inequality constraints in gurobipy (because of the gurobi parsing order syntax)

Take a deep breath and solve the problem step by step.

"""

import json

class Programmer(Agent1):
    def __init__(
        self, model_name="gemma2", solver="gurobipy", debugger_on=True, **kwargs
    ):
        super().__init__(
            name="Programmer",
            description="This is a mathematical programmer agent that is an expert in writing, modifying, and debugging code for optimization problems from the mathematical formulation of the problem. This agent should be called first when a bug or error happens in the code.",
            model_name=model_name,
            **kwargs,
        )

        self._debugger_on = debugger_on
        self.solver = solver

    def generate_reply(self, task: str, state: Dict, sender: Agent) -> Tuple[str, Dict]:
        # add some lines and characters around it to make the input interface nicer
        print("- Programmer agent is called!")
        print()

        if state["solution_status"] == "runtime_error":
            # Enter debugging mode
            bogus_item = None
            for target in ["constraint", "objective", "variables"]:
                for item in state[target]:
                    if not item["status"] in ["coded", "formulated", "runtime_error"]:
                        # raise Exception(
                        #     f"{target} {item} inconsistency in state! \n {json.dumps(state, indent=4)}"
                        # )
                        print(
                            f"{target} {item} inconsistency in state! \n {json.dumps(state, indent=4)}"
                        )
                    if item["status"] == "runtime_error":
                        bogus_item = item
                        break

            if not bogus_item:
                raise Exception(
                    "No runtime error in state!", json.dumps(state, indent=4)
                )

            return self._debug_code(state=state)

        elif state["solution_status"] is None:
            # Enter coding mode
            return self._generate_code_from_formulation(state=state)

        else:
            raise Exception(
                f"Invalid solver_output_status {state['solver_output_status']}!"
            )


    def _debug_code(self, state: Dict) -> Tuple[str, Dict]:
        if not self._debugger_on:
            raise Exception("Debugger is off. Execution failed")

        error_line = None
        bogus_context = None

        for target in ["constraint", "objective", "variables"]:
            for item in state[target]:
                if item["status"] == "runtime_error":
                    bogus_context = item

        context = {}
        prep_code = state["prep_code"]

        if "description" in bogus_context:
            error_line = bogus_context["code"]
            error_message = state["error_message"]
            for parameter in state["parameters"]:
                if parameter["symbol"] in bogus_context["related_parameters"]:
                    prep_code += parameter["code"] + "\n"

            for variable in state["variables"]:
                if variable["symbol"] in bogus_context["related_variables"]:
                    if not "code" in variable:
                        raise Exception(f"Variable {variable} is not coded yet!")

                    prep_code += variable["code"] + "\n"
            prompt = debugging_refined_template_target.format(
                target=target,
                prep_code=prep_code,
                error_line=error_line,
                error_message=error_message,
            )

        elif "definition" in bogus_context:
            error_line = bogus_context["code"]
            error_message = state["error_message"]

            prompt = debugging_refined_template_variable.format(
                target=target,
                prep_code=prep_code,
                error_line=error_line,
                error_message=error_message,
            )

        else:
            raise Exception(
                f"Invalid bogus_context {bogus_context}! \n {json.dumps(state, indent=4)}"
            )

        cnt = 3
        while cnt > 0:
            cnt -= 1
            try:
                print("%^%^%")
                print(prompt)
                response = self.llm_call(prompt=prompt, seed=cnt)
                print(response)
                print("%^%^%")
                response = response[response.find("```json") + 7 :]
                response = response[: response.rfind("```")]

                update = json.loads(response)

                if update["status"] == "correct":
                    bogus_context["status"] = "formulation_error"
                    return update["reason"], state
                elif update["status"] == "fixed":
                    bogus_context["status"] = "coded"
                    bogus_context["code"] = update["fixed_code"]
                    return "The code is fixed! Try evaluating it again.", state
                else:
                    raise Exception(f"Invalid status {update['status']}!")

            except Exception as e:
                print(e)
                print(f"Invalid json format {response}! Try again ...")



    def _generate_code_from_formulation(self, state: Dict) -> Tuple[str, Dict]:
        for variable in state["variables"]:
            print(f"Programming variable {variable['symbol']} ...")

            if variable["status"] == "not_formulated":
                raise Exception(f"Variable {variable} is not formulated yet!")

            elif variable["status"] == "formulated":
                context = {}
                context["definition"] = variable["definition"]
                context["symbol"] = variable["symbol"]
                context["shape"] = variable["shape"]

                messages = [
                    {
                        "role": "system",
                        "content": variable_definition_prompt_templates[0].format(
                            solver=self.solver
                        ),
                    },
                    {
                        "role": "user",
                        "content": variable_definition_prompt_templates[1].format(
                            variable=context,
                        ),
                    },
                ]

                cnt = 3
                while cnt > 0:
                    try:
                        response = self.llm_call(messages=messages, seed=cnt)
                        print(response)
                        code = [
                            r.strip()
                            for r in response.split("=====")
                            if len(r.strip()) > 2
                        ][-1]

                        code = code.strip()
                        while code[0] == "=":
                            code = code[1:].strip()
                        while code[-1] == "=":
                            code = code[:-1].strip()

                        if len(code) < 2:
                            raise Exception(f"Invalid code {code}!")

                        code = code.replace("```python", "").replace("```", "")

                        variable["code"] = code
                        variable["status"] = "coded"
                        break
                    except Exception as e:
                        cnt -= 1
                        import traceback

                        print(traceback.print_exc())
                        print(messages[1]["content"])
                        print(response)
                        print(e)
                        print(f"Invalid response {response}! Try again ...")

                        if cnt == 0:
                            raise e

            elif variable["status"] == "coded":
                pass

        for target in ["constraint", "objective"]:
            for item in state[target]:
                print(f"Programming {target} ...")
                if item["status"] == "not_formulated":
                    raise Exception(f"{target} {item} is not formulated yet!")

                elif item["status"] == "formulated":
                    context = {}
                    context["description"] = item["description"]
                    context["formulation"] = item["formulation"]
                    context["related_variables"] = []
                    context["related_parameters"] = []

                    for parameter in state["parameters"]:
                        if parameter["symbol"] in item["related_parameters"]:
                            context["related_parameters"].append(parameter)

                    for variable in state["variables"]:
                        if variable["symbol"] in item["related_variables"]:
                            if not "code" in variable:
                                raise Exception(
                                    f"Variable {variable} is not coded yet!"
                                )
                            context["related_variables"].append(
                                {
                                    "symbol": variable["symbol"],
                                    "definition": variable["definition"],
                                    "shape": variable["shape"],
                                    "code": variable["code"],
                                }
                            )

                    messages = [
                        {
                            "role": "system",
                            "content": main_prompt_templates[target][0].format(
                                solver=self.solver
                            ),
                        },
                        {
                            "role": "user",
                            "content": main_prompt_templates[target][1].format(
                                context=json.dumps(context, indent=4),
                            ),
                        },
                    ]

                    cnt = 3

                    while cnt > 0:
                        try:
                            response = self.llm_call(messages=messages, seed=cnt)
                            print(response)
                            code = [
                                r.strip()
                                for r in response.split("=====")
                                if len(r.strip()) > 2
                            ][-1]

                            code = code.replace("```python", "").replace("```", "")

                            item["code"] = code
                            item["status"] = "coded"
                            break
                        except Exception as e:
                            import traceback

                            print(traceback.print_exc())
                            print(messages[1]["content"])
                            print(response)
                            cnt -= 1
                            if cnt == 0:
                                raise e

                else:
                    raise Exception(f"{target} {item} is not formulated yet!")

        return "Coding Done! Now we can evaluate the code!", state

from typing import Dict

import json
import traceback

main_prompt_templates = [
    """
You're an expert evaluator in a team of optimization experts. The goal of the team is to solve an optimization problem. Your responsibility is to run the code and evaluate the performance and correctness of the code.
"""
]


prep_code = """
import json
import numpy as np
import math

{solver_prep_code}

with open("{data_json_path}", "r") as f:
    data = json.load(f)

"""


post_code = """

# Get model status
status = model.status

obj_val = None
# check whether the model is infeasible, has infinite solutions, or has an optimal solution
if status == gp.GRB.INFEASIBLE:
    obj_val = "infeasible"
elif status == gp.GRB.INF_OR_UNBD:
    obj_val = "infeasible or unbounded"
elif status == gp.GRB.UNBOUNDED:
    obj_val = "unbounded"
elif status == gp.GRB.OPTIMAL:
    obj_val = model.objVal
"""

class Evaluator(Agent):
    def __init__(self, model_name="gemma2", solver="gurobipy", **kwargs):
        super().__init__(
            name="Evaluator",
            description="This is an evaluator agent that is an expert in running optimization codes, identifying the bugs and errors, and evaluating the performance and correctness of the code.",
            model_name=model_name,
            **kwargs,
        )
        self.solver = solver


    def generate_reply(self, task: str, state: Dict, sender: Agent) -> (str, Dict):
        print("- Evaluator agent is called!")

        res = self._run_code(state=state)

        if not res["success"]:
            state["solution_status"] = "runtime_error"
            state["error_message"] = res["error_message"]
            state["prep_code"] = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )

            if not res["bogus_context"]:
                return f"Bad model! Print DONE to finish the execution.", state

            res["bogus_context"]["status"] = "runtime_error"
            state["solver_output_status"] = res["bogus_context"]["status"]

            return (
                f"There was an error in running the code! {res['error_message']}",
                state,
            )

        else:
            state["solution_status"] = "solved"
            state["solver_output_status"] = res["status"]
            state["obj_val"] = res["obj_val"]
            state["code"] = res["code"]
            return ("Evaluation Done! The problem is solved.", state)


    def _run_code(self, state: Dict):
        local_env = {}
        code = ""
        last_line = ""
        bogus_context = None

        try:
            last_line = prep_code.format(
                solver_prep_code=self.get_solver_prep_code(),
                data_json_path=state["data_json_path"],
            )
            code += last_line + "\n"

            exec(
                last_line,
                local_env,
                local_env,
            )

            for parameter in state["parameters"]:
                if not "code" in parameter:
                    raise Exception(f"Parameter {parameter} is not coded yet!")
                last_line = parameter["code"]
                code += last_line + "\n"
                exec(last_line, local_env, local_env)

            for variable in state["variables"]:
                bogus_context = variable
                last_line = variable["code"]
                code += last_line + "\n"
                exec(last_line, local_env, local_env)

            for constraint in state["constraint"]:
                bogus_context = constraint
                last_line = constraint["code"]
                code += "\n" + last_line + "\n"
                exec(last_line, local_env, local_env)

            bogus_context = state["objective"][0]
            last_line = state["objective"][0]["code"]
            code += "\n" + last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = "OPTIMIZATION CALL"
            last_line = f"\n# Optimize model\nmodel.optimize()\n"
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            bogus_context = None
            last_line = post_code
            code += last_line + "\n"
            exec(last_line, local_env, local_env)

            return {
                "success": True,
                "error_line": None,
                "code": code,
                "obj_val": local_env["obj_val"],
                "status": local_env["status"],
                "error_message": None,
            }

        except Exception as e:
            print("COOOODE")
            print(code)
            print()
            if not bogus_context:
                error_msg = traceback.format_exc()
                raise Exception(
                    f"Unexpected error in running code at {last_line}: "
                    + "\n"
                    + str(e)
                    + "\n\n\n"
                    + error_msg
                )

            error_msg = traceback.format_exc()
            return {
                "success": False,
                "error_line": last_line,
                "code": code,
                "obj_val": None,
                "status": None,
                "error_message": error_msg,
                "bogus_context": bogus_context,
            }

    def get_solver_prep_code(self):
        if self.solver == "gurobipy":
            return "import gurobipy as gp\n\n # Define model\nmodel = gp.Model('model')"
        else:
            raise Exception(f"Solver {self.solver} is not supported yet!")