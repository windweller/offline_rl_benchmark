For each environment, we provide a Gym interface (though in order to stay close to the 
original implementation) and `evaluate_on_environment` function in `env.py` (the main file we work on).

Format guide. Each environment needs to have its own folder. In it `env.py` will define
the APIs for the entire library to use.

**Guide to create a new environment:**

Step 1: Inside `env.py`:
- Implement an `EnvClass` that needs to have a `gym.Env` style environment class (either to serve
as a wrapper or an actual implementation of the environment).
- Implement `load_x_dataset()`, which will be called by `datasets.py`
- Implement `evaluate_on_x_environment()`, which is similar to `d3rlpy.metrics.scorer.evaluate_on_environment`.

Step 2: Inside `datasets.py`:
- Implement `get_x(env_name: str)`, which will download the datasets using a URL request and unpack the zip file. It will 
then hand off these data to `env.py` for processing.

### Sepsis

ArXiv link: 

### TutorBot

Toy environment, very limited state space. Only for quick verification of algorithms.

The transition dynamics is a 4th-order Markov Chain.