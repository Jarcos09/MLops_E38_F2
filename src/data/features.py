import typer
from src.data.preprocess_data import DataPreprocessor
from src.config.config import conf, PROJECT_PATHS, PREPROCESSING_PATHS

app = typer.Typer()

@app.command()
def preprocess():
    PROJECT_PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)

    preprocessor = DataPreprocessor(
        input_path=PREPROCESSING_PATHS.INPUT_FILE,
        output_paths={
            "X_TRAIN": PREPROCESSING_PATHS.X_TRAIN,
            "X_TEST": PREPROCESSING_PATHS.X_TEST,
            "Y_TRAIN": PREPROCESSING_PATHS.Y_TRAIN,
            "Y_TEST": PREPROCESSING_PATHS.Y_TEST
        },
        config={
            "target_columns": conf.preprocessing.target_columns,
            "drop_columns": conf.preprocessing.drop_columns,
            "encoding": {
                "drop": conf.preprocessing.encoding.drop,
                "sparse_output": conf.preprocessing.encoding.sparse_output,
                "handle_unknown": conf.preprocessing.encoding.handle_unknown
            },
            "test_size": conf.preprocessing.test_size,
            "random_state": conf.preprocessing.random_state,
            "target_transform": conf.preprocessing.target_transform
        }
    )
    preprocessor.run()

if __name__ == "__main__":
    app()
