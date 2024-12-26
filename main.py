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
    
    english = input_text.value
    output_div = document.querySelector("#output")
    output_div.innerText = return_text
