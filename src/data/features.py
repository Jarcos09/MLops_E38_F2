import typer
from src.data.preprocess_data import DataPreprocessor
from src.config.config import conf
from src.utils.paths import ensure_path

app = typer.Typer()

@app.command()
def preprocess():
    ensure_path(conf.paths.processed)

    preprocessor = DataPreprocessor(
        input_path=conf.data.interim_data_file,
        output_paths={
            "X_TRAIN": conf.data.processed_data.x_train_file,
            "X_TEST": conf.data.processed_data.x_test_file,
            "Y_TRAIN": conf.data.processed_data.y_train_file,
            "Y_TEST": conf.data.processed_data.y_test_file
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
