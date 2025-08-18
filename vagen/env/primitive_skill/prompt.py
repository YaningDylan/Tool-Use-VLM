def system_prompt(**kwargs):
    """
    Returns the system prompt for the robot arm control.
    Designed for multi-turn interaction with tool-first approach.
    
    Returns:
        str: The system prompt with tool workflow
    """
    return """You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

MULTI-TURN WORKFLOW:
1. FIRST TURN: Observe the image and use tools to understand the scene
2. SUBSEQUENT TURNS: Plan and execute robot actions based on gathered information

Available Tools (for scene understanding):

Basic Environment Tools:
- get_objects(): List all visible objects in the scene
- get_pos(color): Get position of object by color (e.g., get_pos(red))
- get_targets(): Get all target positions in the scene
- get_workspace(): Get robot workspace boundaries

Enhanced YOLO Vision Tools:
- detect_scene(): Use YOLO to detect all objects with detailed visual information and bounding boxes
- find_object(description): Find objects by natural language description (e.g., "red cube on the left")
- get_spatial_relations(): Analyze spatial relationships between detected objects

Robot Actions (for manipulation):
1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Navigation Hints: 
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.
3. The position is the center of the object, when you place, please consider the volume of the object. It's always fine to set z much higher when placing an item.
4. You're facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up.

Example Multi-Turn Interaction:

TURN 1 (Enhanced Scene Understanding):
Human: Put the apple next to the red cube
Image: [shows complex scene with multiple objects]
Assistant: I'll use enhanced vision tools to understand this complex scene.
detect_scene()
find_object("apple")
find_object("red cube")
get_spatial_relations()

TURN 2 (Action Planning):
Enhanced Tool Results:
✓ detect_scene: YOLO detected 6 objects: 2 cubes, 1 apple, 2 targets, 1 bottle
✓ find_object("apple"): Located at pixel (320, 240), env position (45, 60, 25)
✓ find_object("red cube"): Located at pixel (180, 200), env position (70, -30, 20)  
✓ get_spatial_relations: "apple is above and right of red cube"

Now I can plan precise actions: pick the apple and place it next to the red cube.
Assistant: pick(45, 60, 25)|place(90, -30, 25)

TURN 3 (Alternative - Basic Tools):
Human: Move green cube to right target
Image: [simpler scene]
Assistant: I'll use basic tools for this straightforward task.
get_objects()
get_pos(green)
get_targets()

TURN 4 (Action Planning):
Basic Tool Results:
✓ get_objects: ['green_cube', 'left_target', 'right_target']
✓ get_pos(green): (75, 33, 20)
✓ get_targets: {'left_target': (100, -43, 0), 'right_target': (100, 43, 0)}

Now I can execute the action: pick green cube and place at right target.
Assistant: pick(75, 33, 20)|place(100, 43, 50)

Tool Selection Guidelines:
- Use detect_scene() for complex scenes with multiple objects or when you need detailed visual analysis
- Use find_object(description) when looking for specific objects by natural language description
- Use get_spatial_relations() to understand object positions relative to each other
- Use basic tools (get_pos, get_objects, get_targets) for simple, straightforward tasks
- You can combine both enhanced and basic tools in the same turn if needed
"""

def init_observation_template(**kwargs):
    """Template for the very first observation - no tools results yet"""
    observation = kwargs.get("observation")
    instruction = kwargs.get("instruction")
    x_workspace = kwargs.get("x_workspace")
    y_workspace = kwargs.get("y_workspace")
    z_workspace = kwargs.get("z_workspace")
    other_information = kwargs.get("other_information")
    
    # First turn - no object positions provided, force tool use
    return f"""
[Initial Observation]:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}

Available Tools:

Basic Environment Tools:
- get_objects(): List all visible objects in the scene
- get_pos(color): Get position of object by color (e.g., get_pos(red))
- get_targets(): Get all target positions in the scene
- get_workspace(): Get robot workspace boundaries

Enhanced YOLO Vision Tools:
- detect_scene(): Use YOLO to detect all objects with detailed visual information
- find_object(description): Find objects by natural language description (e.g., "red cube on the left")
- get_spatial_relations(): Analyze spatial relationships between detected objects

IMPORTANT: This is the first turn. You must use tools to understand the scene before planning any robot actions!

Tool Selection Tips:
- Use detect_scene() for complex scenes or when you need comprehensive object analysis
- Use find_object(description) when searching for specific objects by description
- Use basic tools for simple, straightforward object queries
- You can combine multiple tools in one turn

Other information:
{other_information}
Decide your next action(s)."""

def action_template(**kwargs):
    """Template for subsequent turns after tool execution"""
    observation = kwargs.get("observation")
    instruction = kwargs.get("instruction")
    x_workspace = kwargs.get("x_workspace")
    y_workspace = kwargs.get("y_workspace")
    z_workspace = kwargs.get("z_workspace")
    valid_actions = kwargs.get("valid_actions")
    other_information = kwargs.get("other_information")
    tool_results = kwargs.get("tool_results", "")
    
    base_template = f"""After your answer, the extracted valid action(s) is {valid_actions}.
After that, the observation is:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}"""

    if tool_results:
        # This is a turn after tool execution - show results and ask for robot actions
        base_template += f"""
Tool Results:
{tool_results}

Now plan your robot actions based on the tool results above.
Available Robot Actions:
- pick(x, y, z): Grasp object at position
- place(x, y, z): Place held object at position  
- push(x1, y1, z1, x2, y2, z2): Push object from one position to another"""
    else:
        # This is a turn without tool results - might need more tools or robot actions
        base_template += """

Available Tools:

Basic Environment Tools:
- get_objects(): List all visible objects in the scene
- get_pos(color): Get position of object by color (e.g., get_pos(red))
- get_targets(): Get all target positions in the scene
- get_workspace(): Get robot workspace boundaries

Enhanced YOLO Vision Tools:
- detect_scene(): Use YOLO to detect all objects with detailed visual information
- find_object(description): Find objects by natural language description
- get_spatial_relations(): Analyze spatial relationships between detected objects

Available Robot Actions:
- pick(x, y, z): Grasp object at position
- place(x, y, z): Place held object at position  
- push(x1, y1, z1, x2, y2, z2): Push object from one position to another"""
    
    base_template += f"""
Other information:
{other_information}
Decide your next action(s)."""
    
    return base_template

def tool_response_template(**kwargs):
    """Template for when tools have been executed and we need robot actions"""
    instruction = kwargs.get("instruction")
    x_workspace = kwargs.get("x_workspace")
    y_workspace = kwargs.get("y_workspace")
    z_workspace = kwargs.get("z_workspace")
    tool_results = kwargs.get("tool_results")
    other_information = kwargs.get("other_information")
    
    return f"""
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}

Tool Results:
{tool_results}

Based on the tool results above, plan your robot actions to complete the instruction.
Available Robot Actions:
- pick(x, y, z): Grasp object at position
- place(x, y, z): Place held object at position  
- push(x1, y1, z1, x2, y2, z2): Push object from one position to another

Other information:
{other_information}
Decide your next action(s)."""

# --- FORMAT_CONFIGS for multi-turn interaction with YOLO ---
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "tool_example": "e.g. <think>I need to understand this complex scene first. I'll use YOLO detection for comprehensive analysis.</think><answer>detect_scene(){action_sep}find_object(\"red cube\"){action_sep}get_spatial_relations()</answer>",
        "action_example": "e.g. <think>Based on the tool results, I need to pick the red cube and place it on the green cube.</think><answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "tool_example": "e.g. <answer>detect_scene(){action_sep}find_object(\"apple\"){action_sep}get_pos(red)</answer>",
        "action_example": "e.g. <answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"
    },
    "grounding": {
        "description": "You should first give your thought process with observation and reasoning, and then your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "tool_example": "<think><observation>I see a complex scene with multiple objects but need to identify their exact positions and relationships.</observation><reasoning>I should use YOLO detection for comprehensive visual analysis, then find specific objects mentioned in the instruction</reasoning></think><answer>detect_scene(){action_sep}find_object(\"red cube\"){action_sep}find_object(\"apple\")</answer>",
        "action_example": "<think><observation>Tool results show red cube at (62,-55,20) and green cube at (75,33,20).</observation><reasoning>I need to pick the red cube and place it on top of the green cube with safe z-height</reasoning></think><answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, and then your answer.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "tool_example": "<think><reasoning>I need to first get comprehensive scene understanding using YOLO, then identify specific objects before planning any actions</reasoning><prediction>After using enhanced vision tools, I will have detailed object positions and spatial relationships to plan the manipulation sequence</prediction></think><answer>detect_scene(){action_sep}find_object(\"red cube\"){action_sep}get_spatial_relations()</answer>",
        "action_example": "<think><reasoning>Based on positions, I should pick red cube first then place it on green cube</reasoning><prediction>After these actions, red cube will be stacked on green cube at position (75,33,50)</prediction></think><answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with observation, reasoning, and prediction, and then your answer.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "tool_example": "<think><observation>I can see multiple objects in the scene but need detailed visual analysis to understand their exact positions and relationships.</observation><reasoning>I must use YOLO detection for comprehensive scene understanding, then find specific objects mentioned in the task</reasoning><prediction>Enhanced vision tools will provide precise coordinates and spatial relationships for planning robot actions</prediction></think><answer>detect_scene(){action_sep}find_object(\"red cube\"){action_sep}find_object(\"apple\")</answer>",
        "action_example": "<think><observation>Red cube is at (62,-55,20), green cube at (75,33,20).</observation><reasoning>I need to stack red on green with safe placement height</reasoning><prediction>Red cube will end up at (75,33,50) on top of green cube</prediction></think><answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"
    }
}

def format_prompt_generator(format_type):
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        action_sep = kwargs.get("action_sep", "|")
        add_example = kwargs.get("add_example", True)
        is_tool_turn = kwargs.get("is_tool_turn", True)  # True for first turn, False for action turns
        
        config = FORMAT_CONFIGS[format_type]
        
        # Build the base prompt text
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}

Your response should be in the format of:
{config["format"]}"""
        
        # Add appropriate example based on turn type
        if add_example:
            if is_tool_turn:
                example = config["tool_example"].format(action_sep=action_sep)
                base_prompt += f"\n\nFor scene understanding: {example}"
            else:
                example = config["action_example"].format(action_sep=action_sep)  
                base_prompt += f"\n\nFor robot actions: {example}"
        
        return base_prompt
    
    return prompt_function

# --- Populate format_prompt dictionary ---
format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS
}