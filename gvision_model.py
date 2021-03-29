import pathlib
import torch
import numpy as np
import gvision_utils
import img_utils

class PretrainedModel():
    def __init__(self, modelname):
        model_pt = model_class_dict[modelname](pretrained=True)
        #model.eval()
        self.model = nn.DataParallel(model_pt.cuda())
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()

    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def forward(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)

    def __call__(self, x):
        return self.predict(x)



class GVisionModel:
    """Returns 2 simulated logits corresponding to correct class and incorrect class, such that attack has signal for optimization"""

    def __init__(self, correct_labelset=['cat'], exp_name="saved_img", save_location="output", loss_margin=1):
        self.labelset = correct_labelset
        self.save_location = save_location
        self.exp_name = exp_name
        self.counter = 0
        self.loss_margin = loss_margin
        self.best_loss = np.inf

        pathlib.Path(save_location).mkdir(parents=True, exist_ok=True)

        with open(f"{save_location}/labels.txt", "w") as f:
            f.write(", ".join(correct_labelset))

    def __call__(self, x):
        return self.predict(x)


    def predict(self, x):
        x = x.cpu().detach().numpy()

        if x.shape[0] != 1:
            raise AssertionError("Batch size must be 1")

        x = x[0]
        results = gvision_utils.gvision_classify_numpy(x)
        print(str(results))

        self.counter += 1

        logits = self._compute_logits(results)

        current_loss = self.loss(y=None, logits=logits)
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            img = img_utils.convert_to_pillow(x)
            img = img_utils.write_text_to_img(img, f"Iter: {self.counter}\n{results}")
            img.save(f"{self.save_location}/{self.exp_name}_{self.counter}.png")

        return torch.tensor(logits)


    def _compute_logits(self, results):
        """Confidence score of the correct and 2nd best class"""
        matching_results = results.match(self.labelset)
        other_results = results.match(self.labelset, inverse=True)


        print('-----------------')
        print("Matching results:")
        print(matching_results)
        

        print("\nNot-matching results:")
        print(other_results)
        print('-----------------')


        print(f"Top label: {matching_results.top_label} - {matching_results.top_score}")
        print(f"2nd best: {other_results.top_label} - {other_results.top_score}\n\n")

        return np.array([[matching_results.top_score + self.loss_margin, other_results.top_score]])

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """

        if logits.shape[0] != 1:
            raise AssertionError("Batch size must be 1")
        if targeted:
            raise AssertionError("Targeted attack should be done by modifying the logit computation procedure")
        if loss_type != 'margin_loss':
            raise AssertionError("Gvision model only supports margin loss") 

        correct, second_best = logits[0]
        return np.array([correct - second_best])