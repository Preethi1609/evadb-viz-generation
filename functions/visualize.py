
import pandas as pd
import os

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


from dataframe.goal_gen import GoalGenerator

class Visualize(AbstractFunction):

    @setup(cacheable=False, function_type="FeatureExtraction", batchable=False)
    def setup(self, config=None):
        if config:
            self.config = config
        else:
            self.config = "spam-ham-label/config_spam_detection.json"

    @property
    def name(self) -> str:
        return "Visualize"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None, 5)],
            ),
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        project_dir = "/home/preethi/projects/evadb-viz-generation/"
        goal_gen = GoalGenerator(df, project_dir)
        summary = goal_gen.summarize(df, file_name="cars.csv")
        print("SUMMARY: ", summary)
        persona = ""
        goals = goal_gen.gen_goals(summary, persona)
        print("GOALS: ", goals)
        goal_gen.visualize(summary, goals, file_name="cars.csv")

        response = ""
        df_dict = {"response": [str(response)]}

        ans_df = pd.DataFrame(df_dict)
        return pd.DataFrame(ans_df)

