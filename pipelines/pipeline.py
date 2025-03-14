from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.container_component
def load_data(
    output_dataset_train: Output[Dataset],
    output_dataset_test: Output[Dataset]
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:latest', 
        command=['python', '/app/data/load_data.py'],
        args=[
            '--output_dataset_train', output_dataset_train.path,
            '--output_dataset_test', output_dataset_test.path
        ]
    )

@dsl.container_component
def train_model(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    trained_model: Output[Model],
    epochs: int = 5
):
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:latest',
        command=['python', '/app/train_model.py'],
        args=[
            '--train_data', train_data.path,
            '--test_data', test_data.path,
            '--trained_model', trained_model.path,
            '--epochs', str(epochs)
        ],
    )
    
@dsl.container_component
def get_latest_model():
    return dsl.ContainerSpec(
        image='timsmans/ml-pipeline:latest',
        command=['python', '/app/get_latest_model.py'],
    )


@dsl.pipeline(
    name="mnist-pipeline",
    description="End-to-end MNIST training pipeline using Docker"
)
def mnist_pipeline(epochs: int = 5):
    # Data loading
    load_data_task = load_data()
    
    # Training
    train_task = train_model(
        train_data=load_data_task.outputs['output_dataset_train'],
        test_data=load_data_task.outputs['output_dataset_test'],
        epochs=epochs
    )

    get_latest_task = get_latest_model()
    
    train_task.set_caching_options(False)
    get_latest_task.set_caching_options(False)

    # Define execution order
    train_task.after(load_data_task)
    get_latest_task.after(train_task)

if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=mnist_pipeline,
        package_path="mnist_training_pipeline.yaml"
    )
