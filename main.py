import networkx as nx
from pyscript import document

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

def genGraph(reg_expr, start_key, G):   # returns end key
    reg_expr = remove_outer_parentheses(reg_expr)
    
    G.add_node(start_key, label = "q") #q
    temp_key = start_key
 
    if isSimple(reg_expr) and "*" not in reg_expr:
        temp_key += 1
        for a, s in enumerate(reg_expr):
            if s == "e" : s = "\u03B5"
            G.add_node(temp_key, label = "") 
            G.add_edge(temp_key - 1, temp_key, label = s)
            temp_key += 1
        G.add_node(temp_key - 1, label = "f") 
        return temp_key - 1
    
    parts = split_reg_expr(reg_expr)
    print(parts)
    if len(parts) == 1:
        if "(" in reg_expr or ")" in reg_expr:
            before, paren, after = processParenthesis(reg_expr)
        else:
            kleene_index = reg_expr.index('*')
            before = reg_expr[:kleene_index - 1]
            paren = reg_expr[kleene_index - 1]
            after = reg_expr[kleene_index:] 
        print(before, paren, after)
        if len(before) > 0:
            end_key = genGraph(before, temp_key + 1, G)
            G.add_edge(start_key, temp_key + 1, label = "\u03B5")
            temp_key = end_key
            start_key = end_key

        end_key = genGraph(paren, temp_key + 1, G)    
        if after.startswith("*"):       
            G.add_edge(temp_key, temp_key + 1, label = "\u03B5") # starting edge
            G.add_edge(end_key, temp_key, label = "\u03B5")  # looping back edge    (end_key -> temp_key + 1)
            temp_key = end_key + 1
            G.add_node(temp_key, label = "f")
            G.add_edge(start_key, temp_key, label = "\u03B5")  # direct edge
            G.add_edge(end_key, temp_key, label = "\u03B5") 
            after = after[1:]
            end_key = temp_key

        if len(after) > 0:
            end_key = genGraph(after, temp_key + 1, G)
            G.add_edge(temp_key, temp_key + 1, label = "\u03B5")
            temp_key = end_key
        return end_key

    end_keys = []
    for part in parts:
          end_key = genGraph(part, temp_key + 1, G)
          G.add_edge(start_key, temp_key + 1, label = "\u03B5")
          temp_key = end_key
          end_keys.append(end_key)
    
    temp_key = max(end_keys) + 1
    G.add_node(temp_key, label = "f") #f
    for end_key in end_keys:
        G.add_edge(end_key, temp_key, label = "\u03B5")
    return temp_key

def checkIfAccepted(G, s, endKey):
   
    def getEpsilonClosure(state):   # gets all epsilon reachable states from current state
        closure = [state]
        stack = [state]
        visited = {state: False for state in G.nodes}

        while stack:
            curr = stack.pop()
            visited[curr] = True
            for neighbor in G.neighbors(curr):
                if not visited[neighbor]:             
                    edge = G.edges[curr, neighbor]["label"]
                    if edge == "\u03B5":
                        closure.append(neighbor)
                        stack.append(neighbor)
        return closure
    EpsilonDict = {state: getEpsilonClosure(state) for state in G.nodes}    # pre-compute epsilon closure for all states
    
    if s == "e": return endKey in EpsilonDict[0]
    stack = []
    stack.append((0, 0))     # key, and index achieved 
    while stack:
        curr, index = stack.pop()
        if (curr == endKey or endKey in EpsilonDict[curr]) and index == len(s): return True
        for state in EpsilonDict[curr]:
            for neighbor in G.neighbors(state):  
                edge = G.edges[state, neighbor].get("label", None)
                if s[index:].startswith(edge):
                    stack.append((neighbor, index + len(edge)))
       
        for neighbor in G.neighbors(curr):  
            edge = G.edges[curr, neighbor].get("label", None)
            if s[index:].startswith(edge):
                stack.append((neighbor, index + len(edge)))
    return False

def NFAtoDFA(G):
    def getEpsilonClosure(state):   # gets all a- reachable states from current state (reachable to the state but not beyond)
        closure = {state}
        stack = [state]
        visited = {state: False for state in G.nodes}

        while stack:
            curr = stack.pop()
            visited[curr] = True
            for neighbor in G.neighbors(curr):
                if not visited[neighbor]:             
                    edge = G.edges[curr, neighbor]["label"]
                    if edge == "\u03B5":
                        closure.add(neighbor)
                        stack.append(neighbor)
        return closure
    EpsilonDict = {state: getEpsilonClosure(state) for state in G.nodes}    # pre-compute epsilon closure for all states
    
    H = nx.MultiDiGraph()
    start_state = tuple(sorted(EpsilonDict[0]))
    stack = [start_state]
    
    H.add_node(start_state)  #start node of H
    H.nodes[start_state]["is_start"] = True

    def getAlphabet(G):
        alphabet = set()
        for edge in G.out_edges(data = True):
            if edge[2]["label"] != "\u03B5":
                alphabet.add(edge[2]["label"])
        return alphabet
    alphabet = getAlphabet(G)
    
    while stack:
        curr_state = stack.pop()
        for a in alphabet:
            new_state = set()
            for state in curr_state:
                for edge in G.out_edges(state, data = True):
                    if a == edge[2]["label"]:
                        new_state.update(EpsilonDict[edge[1]])
            if not any(data["label"] == a for _, _, data in H.edges(tuple(sorted(curr_state)), data=True)):
                H.add_edge(tuple(sorted(curr_state)), tuple(sorted(new_state)), label = a)
                if len(new_state) != 0:
                    stack.append(tuple(sorted(new_state)))
            # if len(new_state) == 0:
            #     for a in alphabet:
            #         H.add_edge(tuple(sorted(new_state)), tuple(sorted(new_state)), label = a)
    
    if () in H.nodes():
        for a in alphabet:
            H.add_edge((), (), label = a)

    max_key = max(G.nodes)
    for node in H.nodes():
        if max_key in node:
            H.nodes[node]["is_end"] = True
                
    return H

def checkIfEquivalent(reg1, reg2):
    if reg1 == "" or reg2 == "" and reg1 != reg2: return False
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    genGraph(reg1, 0, G1)
    genGraph(reg2, 0, G2)
    D1 = NFAtoDFA(G1)
    D2 = NFAtoDFA(G2)
  
    def getAlphabet(G):
        alphabet = set()
        for edge in G.out_edges(data = True):
            if edge[2]["label"] != "\u03B5":
                alphabet.add(edge[2]["label"])
        return alphabet
    if getAlphabet(G1) != getAlphabet(G2):
        return False
    alphabet = getAlphabet(G1)

    Prod = nx.MultiDiGraph()
    start1 = ()
    start2 = ()
    end1 = []
    end2 = []
    for g, g_data in D1.nodes(data=True):
        for h, h_data in D2.nodes(data=True):
            if g_data.get("is_start", False):
                start1 = g
            if h_data.get("is_start", False):
                start2 = h
            if g_data.get("is_end", False):
                end1.append(g)
            if h_data.get("is_end", False):
                end2.append(h)

            Prod.add_node((g,h))
            
   
    for node in Prod.nodes():
        g, h = node     # g is D1, h is D2
        for a in alphabet:
            for _, neighbor, edge_attr in D1.out_edges(g, data= True):
                if edge_attr["label"] == a:
                    g_r = neighbor
            for _, neighbor, edge_attr in D2.out_edges(h, data= True):
                if edge_attr["label"] == a:
                    h_r = neighbor    
            Prod.add_edge((g,h), (g_r, h_r), label = a)

    start = (start1, start2)
    Prod.nodes[start]["is_start"] = True

    stack = [start]
    visited = set([start])

    while stack:
        curr = stack.pop()  # curr[0] has D1 node, curr[1] has D2 node
        visited.add(curr)

        if (curr[0] in end1) != (curr[1] in end2):
            return False

        for neighbor in Prod.neighbors(curr):
            if neighbor not in visited:
                stack.append(neighbor)

    return True

def process_regex(event):
    input_text = document.querySelector("#regex")
    input_texto = document.querySelector("#input_string")
    G = nx.DiGraph()
    endkey = genGraph(input_text.value, 0, G)

    ret = checkIfAccepted(G, input_texto.value, endkey)
    return_text = ""
    if ret:
        return_text = "True"
    else:
        return_text = "False"
    
    output_div = document.querySelector("#output")
    output_div.innerText = return_text

def check_equivalence(event):
    reg1 = document.querySelector("#regex1")
    reg2 = document.querySelector("#regex2")
    
    ret = checkIfEquivalent(reg1.value, reg2.value)
    return_text = ""
    if ret:
        return_text = "Equivalent"
    else:
        return_text = "Not Equivalent"

    output_div = document.querySelector("#output2")
    output_div.innerText = return_text