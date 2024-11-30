from pyscript import document

import networkx as nx
import matplotlib.pyplot as plt
import re

def isSimple(s):
    return "(" not in s and ")" not in s and "/" not in s and "+" not in s

def remove_outer_parentheses(s):
    if s is None:
        return None
    
    def is_balanced(string):
        """Check if a string has balanced parentheses."""
        stack = []
        for char in string:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if not stack:
                    return False
                stack.pop()
        return not stack

    # Keep removing outer parentheses only if they are balanced
    while s.startswith("(") and s.endswith(")") and is_balanced(s[1:-1]):
        s = s[1:-1]
    return s

def split_reg_expr(expr):
    parts = []
    current_part = []
    paren_depth = 0  # Track the depth of nested parentheses

    # Loop through the characters of the expression
    for char in expr:
        if char == '(':
            paren_depth += 1  # Entering a parenthesis block
            current_part.append(char)
        elif char == ')':
            paren_depth -= 1  # Exiting a parenthesis block
            current_part.append(char)
        elif char == '+' and paren_depth == 0:
            # Split at the '+' only if we are not inside parentheses
            parts.append(''.join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)

    # Add the last part to the list
    parts.append(''.join(current_part).strip())

    return parts

def processParenthesis(s):
    if s is None: return s, None, None
    stack = []
    start = 0
    result = []
   
    # Step through the string to find matching parentheses
    for i, char in enumerate(s):
        if char == '(':
            if not stack:
                start = i  # Mark the start of the first parentheses
            stack.append(char)
        elif char == ')':
            stack.pop()
            if not stack:  # If stack is empty, we matched the outermost parentheses
                result.append(s[start:i+1])
            
    
    if result:
        before_parentheses = s.split(result[0])[0]
        first_parentheses = result[0]
        after_parentheses = s[s.find(first_parentheses) + len(first_parentheses):]
        return before_parentheses, first_parentheses, after_parentheses
    else:
        return s, None, None

def crush_epsilon(reg_expr):
    result = []
    for i, char in enumerate(reg_expr):
        if char == 'e':
            if i > 1 and reg_expr[i-1] in ["(", "+"]:
                if i < len(reg_expr) - 1 and reg_expr[i+1] in [")", "+"]:
                    result.append(char)
        else: result.append(char)
    return ''.join(result)

def parse(reg_expr):
    G = nx.DiGraph()
    reg_expr = crush_epsilon(reg_expr)
   # print(reg_expr)
    G,none, endkeys = genGraph(reg_expr, 0, G)
   # print(endkeys)
    drawGraph(G)
    return G, endkeys
    
def genGraph(reg_expr, prev_key, G):
    reg_expr = remove_outer_parentheses(reg_expr)
    if isSimple(reg_expr) and "*" not in reg_expr:
        G.add_node(prev_key, label = [reg_expr, "1"])
        return G, prev_key + 1, [prev_key]      # return Graph, last used key (such tthat it is safe for the next call to use said key) and list of end keys
    if isSimple(reg_expr) and "*" in reg_expr:
        end_key = prev_key
        str = reg_expr
        while "*" in str:
            kleene_index = str.index('*')
            before_kleene = str[:kleene_index - 1]
            kleene = str[kleene_index - 1]

            str = str[kleene_index + 1:]
            if len(before_kleene) > 0:
                G.add_nodes_from([(prev_key, {'label': [before_kleene, "2"]}), 
                  (prev_key + 1, {'label': ["*", "3"]}), 
                  (prev_key + 2, {'label': [kleene, "3"]})])

                G.add_edges_from([(prev_key, prev_key + 1),
                              (prev_key + 1, prev_key + 2),
                              (prev_key + 2, prev_key + 1)])
                end_key = prev_key + 1
                if len(str) > 0:
                    G.add_node(prev_key + 3, label = [str, "2"])
                    G.add_edge(prev_key + 1, prev_key + 3)
                    end_key = prev_key + 3
                prev_key += 3
            else:
                G.add_nodes_from([
                  (prev_key, {'label': ["*", "3"]}), 
                  (prev_key + 1, {'label': [kleene, "3"]})])
                G.add_edges_from([(prev_key, prev_key + 1),
                              (prev_key + 1, prev_key)])
                end_key = prev_key
                if len(str) > 0:
                    G.add_node(prev_key + 2, label = [str, "2"])
                    G.add_edge(prev_key, prev_key + 2)
                    end_key = prev_key + 2
                prev_key += 2
        if len(str) > 0:
            return G, prev_key + 1, [end_key] 
        else: return G, prev_key, [end_key] 
    # so far simple cases w/out () or + handled.
    parts = split_reg_expr(reg_expr)
    if len(parts) == 1:
        # need to handle more complex cases

        if "(" in reg_expr or ")" in reg_expr:
    
            before, paren, after = processParenthesis(reg_expr)
            print(before)
            print(paren)
            print(after)
            if len(before) > 0:
                none, prev_key, end_keys = genGraph(before, prev_key, G)
                #print(end_keys)
                print(prev_key)
                #return G, prev_key, end_keys
                G.add_node(prev_key, label = ["temp", "temp"])
                G.add_edge(end_keys[0], prev_key)

            
            if after.startswith("*"):
                old_prev_kleene = prev_key
                G.add_node(prev_key, label = ["*", "3.1"])
                G.add_node(prev_key + 1, label = ["temp", "temp"])
                G.add_edge(prev_key, prev_key + 1)
                prev_key += 1

            none, prev_key, end_keys = genGraph(paren, prev_key, G)
            #print(end_keys) 
            #print("end keys ^")
            if after.startswith("*"):
                after = after[1:]
                print(old_prev_kleene)
                for e in end_keys:
                    G.add_edge(e, old_prev_kleene)
                end_keys = [old_prev_kleene]

            old_prev = prev_key
            if len(after) > 0:
                none, prev_key, end_keys2 = genGraph(after, prev_key, G)
                #print(end_keys)
                for e in end_keys:
                    G.add_edge(e, old_prev)
                end_keys = end_keys2

            return G, prev_key, end_keys
        
    end_keys = []
    root_key = prev_key
    G.add_node(root_key, label = ["root", "root"])
    prev_key += 1

    for part in parts:
        old_prev = prev_key
        none, prev_key, end_key = genGraph(part, prev_key, G)

        G.add_edge(root_key, old_prev)
        prev_key += 1
        end_keys.extend(end_key)

    return G, prev_key, end_keys

from collections import defaultdict

def checkIfAccepted(G, s, endkeys): #NOT WORKING to see if something is NOT in the language (use recursion?)
    stack = []    # start key, and index of string seen
    visited = defaultdict(int)
    str = G.nodes[0]["label"][0]
    if str == "root" or str == "*" or str == "e":
        stack.append((0, 0))
    else:
        if s.startswith(str):
            stack.append((0, len(str)))
        else:
            return False
    while stack:
        #print(len(stack))
        curr, index = stack.pop()
        visited[curr]+=1

        if visited[curr] > len(s) + 1:
            return False

        #print(G.nodes[curr]["label"][0])
        if curr in endkeys: canEnd = True
        else: canEnd = False

        #s = "e" if epsilon
        if s == "e" and G.nodes[curr]["label"][0] in ["e", "*",""] and canEnd:
            return True
        if len(s) == index and canEnd:
            return True

        for neighbor in G.neighbors(curr):
           str = G.nodes[neighbor]["label"][0]
           if str == "root" or str == "*" or str == "e":
                stack.append((neighbor, index))
           else:
                if s[index:].startswith(str):
                    index += len(str)
                    stack.append((neighbor, index))
        

    return False

def drawGraph(G):
    # Layout
    pos = nx.shell_layout(G)

    # Generate combined labels with keys
    combined_labels = {
        node: f"{node}:{data.get('label', [None])[0]}" 
        for node, data in G.nodes(data=True)
    }
    
    node_colors = []
    for node, data in G.nodes(data=True):
        if "label" in data and len(data["label"]) > 1 and data["label"][1] == "kleene_child":
            node_colors.append("orange")  # Highlight kleene_child nodes in orange
        else:
            node_colors.append("lightblue")  # Default color for other nodes


    # Draw graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size = 1000)
    nx.draw_networkx_labels(G, pos, combined_labels, font_size=12)
   
    plt.show()

#G, endkeys = parse("(a+b*+c)*")
#print(endkeys)
# Example string
#s = "abbc"
#print(checkIfAccepted(G, s, endkeys))

#parts = split_reg_expr(s)
#print(parts)

#print(crush_epsilon("1+(1+e)+e+eeeee(1)"))

regex_pattern = ""

def update_regex(event):
    global regex_pattern
    input_field = document.querySelector("#regex")
    regex_pattern = input_field.value

def validate_input(event):
    input_field = document.querySelector("#input-text")
    input_text = input_field.value
    result_div = document.querySelector("#validation-result")
  
        # Try to compile the regex pattern and match with input string
    G, endkeys = parse(regex_pattern)
    res = checkIfAccepted(G, s, endkeys)
        #pattern = re.compile(regex_pattern)
    if res:
        result_div.innerText = f"✅ Input matches the regex"
    else:
        result_div.innerText = f"❌ Input does NOT match the regex"
  
        
# def translate_english(event):
#     input_text = document.querySelector("#english")
#     english = input_text.value
#     output_div = document.querySelector("#output")
#     output_div.innerText = "JHIIIERFJRF"
