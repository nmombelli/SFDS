import pandas as pd
import os

from lime import submodular_pick


def xai_lime_local(lime_explainer, data_row, model, cust_id, **kwargs):

    exp = lime_explainer.explain_instance(
        data_row=data_row,
        predict_fn=model.predict_proba,
        num_features=6,
        **kwargs
    )

    exp.save_to_file(file_path=f"{os.environ['PATH_OUT_LIME']}/{cust_id}_LIME.html")

    return


def xai_lime_global(lime_explainer, X_test, model):

    # SP-LIME returns explanations on a sample set to provide a non-redundant global decision boundary of the model
    sp_obj = submodular_pick.SubmodularPick(
        explainer=lime_explainer,
        data=X_test.values,
        predict_fn=model.predict_proba,
        num_features=5,
        num_exps_desired=10
    )

    # Open the file in write mode
    with open(f"{os.environ['PATH_OUT_LIME']}/GLOBAL_LIME.html", "w", encoding="utf-8") as file:
        # Write the initial part of the HTML content
        file.write(
            """
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CHURN: LIME GLOBAL xAI</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                .gallery {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .gallery img {
                    max-width: 100%;
                    height: auto;
                }
            </style>
            </head>
            <body>
            <h1>CHURN: LIME GLOBAL xAI</h1>
            <div class="gallery">
            """
        )

        # Iteratively append images
        for exp in sp_obj.sp_explanations:

            # associate Customer's id since the native sp_obj does not do this! SHAME ON YOU LIME!
            # an approximation is used since LIME makes round values.
            ser_tmp = pd.Series(dict(zip(exp.domain_mapper.feature_names, exp.domain_mapper.feature_values)))
            ser_tmp = ser_tmp.astype(float)
            cust_id = 'Invalid Match (due to LIME approximations)'
            for i in X_test.index:
                if round(X_test.loc[i], 2).equals(ser_tmp):
                    cust_id = str(i)

            html_tmp = exp.as_html()
            file.write(f'<div <p><b></br>CUSTOMER: {cust_id}</b></p> </br> {html_tmp}</div>')

        # Write the closing part of the HTML content
        file.write(
            """
            </div>
            </body>
            </html>
            """
        )

    return


if __name__ == '__main__':

    print('I AM READY')
    # xai_lime()
