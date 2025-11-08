# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
)

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import load_image
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, calculate_rouge_l

DEFAULT_PROMPTS = [
    "Apple is red. Banana is Yellow. " * 800 + "Apple is",
    "The capital of the United Kingdom is",
    "Today is a sunny day and I like",
    "AI is a field of computer science focused on",
    # the output of gemma-2-2b from SRT is unstable on the commented prompt
    # "The capital of France is",
]
TEST_RERANK_QUERY_DOCS = [
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin is well known for its museums.",
        ],
    },
    {
        "query": "How many people live in Berlin?",
        "documents": [
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
            "Berlin is well known for its museums.",
        ],
    },
]

dirpath = os.path.dirname(__file__)
file_path = os.path.join(dirpath, "long_prompt.txt")
if os.path.exists(file_path) and os.path.isfile(file_path):
    with open(file_path, "r") as f:
        long_prompt = f.read()
else:
    long_prompt = 'You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\nIntroduction\n\nThroughout U.S. history, Congress has created advisory commissions to assist in the development of public policy. Among other contexts, commissions have been used following crisis situations, including the September 11, 2001, terrorist attacks and the 2008 financial crisis. In such situations, advisory commissions may potentially provide Congress with a high-visibility forum to assemble expertise that might not exist within the legislative environment; allow for the in-depth examination of complex, cross-cutting policy issues; and lend bipartisan credibility to a set of findings and recommendations.\nAs Congress considers its range of responses to the coronavirus pandemic, the creation of one or more congressional advisory commissions is an option that could provide a platform for evaluating various pandemic-related policy issues over time. Past congressional advisory commissions have retrospectively evaluated policy responses, brought together diverse groups of experts, and supplemented existing congressional oversight mechanisms. Policymakers may determine that creating an advisory commission is unnecessary and instead prefer to utilize existing congressional oversight structures, such as standing or select committees, or already established oversight entities.\nThis report provides a comparative analysis of five proposed congressional advisory commissions that would investigate various aspects of the COVID-19 pandemic. The five proposed commissions are found in H.R. 6429 (the National Commission on COVID-19 Act, sponsored by Representative Stephanie Murphy), H.R. 6431 (the Made in America Emergency Preparedness Act, sponsored by Representative Brian Fitzpatrick), H.R. 6440 (the Pandemic Rapid Response Act, sponsored by Representative Rodney Davis), H.R. 6455 (the COVID-19 Commission Act, sponsored by Representative Bennie Thompson), and H.R. 6548 (the National Commission on the COVID-19 Pandemic in the United States Act, sponsored by Representative Adam Schiff). The overall structures of each of the proposed commissions are similar in many respects, both to each other and to previous independent advisory entities established by Congress. Specifically, the proposed commissions would (1) exist temporarily; (2) serve in an advisory capacity; and (3) report a work product detailing the commission\'s findings, conclusions, and recommendations. That said, each particular proposed commission has distinctive elements, particularly concerning its membership structure, appointment structure, and time line for reporting its work product to Congress.\nThis report compares the (1) membership structure, (2) appointment structure, (3) rules of procedure and operation, (4) duties and reporting requirements, (5) powers of the commission, (6) staffing issues, and (7) funding for each of the proposed COVID-19 commissions. Table 1 (at the end of this report) provides a side-by-side comparison of major provisions of the five proposals.\n\n	Membership Structure\n\nSeveral matters related to a commission\'s membership structure might be considered. They include the size of a commission, member qualifications, compensation of commission members, and requirements for partisan balance. \n\n		Size of Commission\n\nIn general, there is significant variation in the size of congressional advisory commissions. Among 155 identified congressional commissions created between the 101 st Congress and the 115 th Congress, the median size was 12 members, with the smallest commission having 5 members and the largest 33 members.\nThe membership structure of each of the five proposed commissions is similar to previous independent advisory entities created by Congress. H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 would each create a 10-member entity. H.R. 6455 would create a 25-member entity.\n\n		Qualifications\n\nPast legislation creating congressional commissions has often required or suggested that commission members possess certain substantive qualifications. Such provisions arguably make it more likely that the commission is populated with genuine experts in the policy area, which may improve the commission\'s final work product.\nH.R. 6455 would provide that commissioners "shall be a United States person with significant expertise" in a variety of fields related to public health and public administration. H.R. 6440 , H.R. 6429 , H.R. 6431 , and H.R. 6548 would provide "the sense of Congress" that commission members should be "prominent U.S. citizens" who are nationally recognized experts in a variety of fields relevant to the pandemic and response efforts. In addition, H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 all prohibit the appointment of federal, state, and local government employees and officers. H.R. 6455 would prohibit federal employees from being commission members.\n\n		Compensation of Commission Members\n\nSome congressional commissions have compensated their members. For example, the National Commission on Terrorist Attacks Upon the United States (9/11 Commission) and the Financial Crisis Inquiry Commission provided that commission members could be compensated at a daily rate of basic pay. Nearly all have reimbursed members for travel expenses. Those that have provided for commissioner compensation most frequently provided compensation at the daily equivalent of level IV of the Executive Schedule.\nEach of the five proposals would provide that commission members be compensated at a rate "not to exceed the daily equivalent of the annual rate of basic pay" for level IV of the Executive Schedule, "for each day during which that member is engaged in the actual performance of duties of the Commission." Members of three proposed commissions would receive travel expenses, including a per diem.\n\n		Partisan Limitations\n\nEach proposal provides a limit on the number of members appointed from the same political party. H.R. 6455 would provide that not more than 13 of its 25 members may be from the same party. H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 would provide that not more than 5 (of 10) members are from the same party. Most previous advisory entities created by Congress do not impose formal partisan restrictions on the membership structure. It may also be difficult to assess the political affiliation of potential members, who may have no formal affiliation (voter registration, for example) with a political party. Instead, most past advisory commissions usually achieve partisan balance through the appointment structure; for instance, by providing equal (or near-equal) numbers of appointments to congressional leaders of each party.\n\n	Appointment Structure\n\nPast congressional commissions have used a wide variety of appointment structures. Considerations regarding appointment structures include partisan balance, filling vacancies, and the time line for making commission appointments.\nThe statutory scheme may directly designate members of the commission, such as a specific cabinet official or a congressional leader. In other cases, selected congressional leaders, often with balance between the parties, appoint commission members. A third common statutory scheme is to have selected leaders, such as committee chairs and ranking members, recommend candidates for appointment to a commission. These selected leaders may act either in parallel or jointly, and the recommendation may be made either to other congressional leaders, such as the Speaker of the House and President pro tempore of the Senate, or to the President.\nEach of the five commission proposals would delegate most or all appointment authority to congressional leaders (including chamber, party, and committee leaders; see Table 1 for details). Additionally, H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 provide for one appointment to be made by the President. H.R. 6429 , H.R. 6431 , and H.R. 6548 would have the President appoint the commission\'s chair. H.R. 6455 has its membership appointed by the chairs and ranking members of designated House and Senate committees, and the Joint Economic Committee. H.R. 6455 does not provide any executive branch appointments.\nAttention to the proper balance between the number of members appointed by congressional leaders and by other individuals (such as the President), or to the number of Members of Congress required to be among the appointees, or to the qualifications of appointees, can be significant factors in enabling a commission to fulfill its congressional mandate.\nIn general, a commission\'s appointment scheme can impact both the commission\'s ability to fulfill its statutory duties and its final work product. For instance, if the scheme provides only for the appointment of Members of Congress to the commission, it arguably might not have the technical expertise or diversity of knowledge to complete its duties within the time given by statute. Similarly, if the appointment scheme includes qualifying provisos so specific that only a small set of private citizens could serve on the panel, the commission\'s final work product may arguably only represent a narrow range of viewpoints. None of the proposed COVID-19 commissions specify whether Members of Congress may serve on the commission.\n\n		Partisan Balance in Appointment Authority\n\nMost previous congressional advisory commissions have been structured to be bipartisan, with an even (or near-even) split of appointments between leaders of the two major parties. By achieving a nonpartisan or bipartisan character, congressional commissions may make their findings and recommendations more politically acceptable to diverse viewpoints. The bipartisan or nonpartisan arrangement can give recommendations strong credibility, both in Congress and among the public, even when dealing with divisive public policy issues. Similarly, commission recommendations that are perceived as partisan may have difficulty gaining support in Congress.\nIn some cases, however, bipartisanship also can arguably impede a commission\'s ability to complete its mandate. In situations where a commission is tasked with studying divisive or partisan issues, the appointment of an equal number of majority and minority commissioners may serve to promote partisanship within the commission rather than suppress it, raising the possibility of deadlock where neither side can muster a majority to act.\nEach of the five proposals employs a structure where leaders in both the majority and minority parties in Congress would make appointments. H.R. 6429 , H.R. 6431 , and H.R. 6548 would provide for five majority and five minority appointments, including one for the President. H.R. 6440 would include two each by the Senate majority leader, the Senate minority leader, and the Speaker of the House, with one appointment by the House minority leader and one by the President, and the chair appointed by the Speaker and vice chair appointed by the Senate majority leader. H.R. 6455 would have 12 majority and 12 minority appointments made by the 12 committee chairs and ranking members and one member jointly appointed by the chair and vice chair of the Joint Economic Committee.\n\n		Vacancies\n\nAll five proposals provide that vacancies on the commission will not affect its powers and would be filled in the same manner as the original appointment.\n\n		Deadline for Appointments\n\nThree of the bills propose specific deadlines for the appointment of commissioners. H.R. 6429 and H.R. 6548 provide that appointments are made between specific dates in January or February 2021. Further, H.R. 6429 provides that commission members could be appointed in September 2020, if there is no longer a COVID-19 public health emergency in effectâas determined by the Secretary of Health and Human Servicesâas of August 31, 2020. H.R. 6440 would require all appointments be made by December 15, 2020. H.R. 6455 would require appointments to be made within 45 days after enactment. H.R. 6429 , H.R. 6440 , and H.R. 6548 would start the commission\'s work in early 2021, as the commission cannot operate without the appointment of members. H.R. 6429 , however would provide that the proposed commission\'s work would begin no later than October 31, 2020, if members are appointed in September 2020. H.R. 6431 does not specify a deadline for the appointment of members.\nTypically, deadlines for appointment can range from several weeks to several months. For example, the deadline for appointments to the Antitrust Modernization Commission was 60 days after the enactment of its establishing act. The deadline for appointment to the Commission on Wartime Contracting in Iraq and Afghanistan was 120 days from the date of enactment. The deadline for appointment to the 9/11 Commission was December 15, 2002, 18 days after enactment of the act.\n\n	Rules of Procedure and Operations\n\nWhile most statutes that authorize congressional advisory commissions do not provide detailed procedures for how the commission should conduct its business, the statutory language may provide a general structure, including a mechanism for selecting a chair and procedures for creating rules. None of the five COVID-19 commission proposals contain language that directs the process for potentially adopting rules of procedure. For a comparison of each proposed commission\'s specified rules of procedures and operations, see Table 1 .\n\n		Chair Selection\n\nEach bill provides for the selection of a chair and/or vice chair of the commission. H.R. 6429 , H.R. 6431 , and H.R. 6548 would have the chair appointed by the President and the vice chair appointed by congressional leaders of the political party opposite the President. H.R. 6440 would have the chair appointed by the Speaker of the House (in consultation with the Senate majority leader and the House minority leader) and the vice chair appointed by the Senate majority leader (in consultation with the Speaker of the House and the Senate minority leader). H.R. 6455 would have the chair and vice chair chosen from among commission members by a majority vote of the commission, and would require the chair and vice chair to have "significant experience" in areas to be studied by the commission.\n\n		Initial Meeting Deadline\n\nAs with the timing of commission appointments, some authorizing statutes are prescriptive in when the commission\'s first meeting should take place. Three of the bills analyzed here provide specific time lines for the commission\'s first meeting. H.R. 6429 would require the first meeting to be no later than March 15, 2021, unless members are appointed in September 2020 (if no public health emergency exists). H.R. 6455 would require the first meeting within 45 days after the appointment of all commission members, which isâgiven the 45-day deadline for appointmentâeffectively a maximum of 90 days after enactment. H.R. 6548 would direct the commission to hold its initial meeting "as soon as practicable," but not later than March 5, 2021. H.R. 6431 and H.R. 6440 do not provide for an initial meeting deadline. Instead, they direct the commission to meet "as soon as practicable." \n\n		Quorum\n\nMost commission statutes provide that a quorum will consist of a particular number of commissioners, usually a majority, but occasionally a supermajority. All five bills would provide for a quorum requirement. H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 would define a quorum as 6 (of 10) members. H.R. 6455 would provide that a quorum is 18 of 25 members (72%).\n\n		Public Access\n\nAll five commission bills would require commission meetings to be open to the public. Each bill would also require that reports be made publicly available.\n\n		Formulating Other Rules of Procedure and Operations\n\nAbsent statutory guidance (eithe r in general statutes or in individual statutes authorizing commissions), advisory entities vary widely in how they adopt their rules of procedure. In general, three models exist: formal written rules, informal rules, and the reliance on norms. Any individual advisory entity might make use of all three of these models for different types of decisionmaking. \nThe choice to adopt written rules or rely on informal norms to guide commission procedure may be based on a variety of factors, such as the entity\'s size, the frequency of meetings, member preferences regarding formality, the level of collegiality among members, and the amount of procedural guidance provided by the entity\'s authorizing statute. Regardless of how procedural issues are handled, protocol for decisionmaking regarding the following operational issues may be important for the commission to consider at the outset of its existence: eligibility to vote and proxy rules; staff hiring, compensation, and work assignments; hearings, meetings, and field visits; nonstaff expenditures and contracting; reports to Congress; budgeting; and procedures for future modification of rules. None of the five COVID-19 commission proposals specify that the proposed commission must adopt written rules.\n\n		FACA Applicability\n\nThe Federal Advisory Committee Act (FACA) mandates certain structural and operational requirements, including formal reporting and oversight procedures, for certain federal advisory bodies that advise the executive branch. Three proposals ( H.R. 6429 , H.R. 6431 , and H.R. 6548 ) specifically exempt the proposed commission from FACA. Of the remaining two, FACA would also likely not apply to the commission proposed in H.R. 6455 because it would be appointed entirely by Members of Congress, although it only specifies that its final report is public, not whether it is specifically sent to Congress and/or the President. It is not clear that FACA would apply to the commission proposed in H.R. 6440 . Although it includes a presidential appointment and its report would be sent to both Congress and the President, its establishment clause specifies that the commission "is established in the legislative branch," and a super-majority of its members would be appointed by Congress.\n\n	Duties and Reporting Requirements\n\nMost congressional commissions are generally considered policy commissionsâtemporary bodies that study particular policy problems and report their findings to Congress or review a specific event. \n\n		General Duties\n\nAll five of the proposed commissions would be tasked with duties that are analogous to those of past policy commissions. While the specific mandates differ somewhat, all proposed commissions are tasked with investigating aspects of the COVID-19 pandemic and submitting one or more reports that include the commission\'s findings, conclusions, and recommendations for legislative action. H.R. 6440 would specifically require the commission to avoid unnecessary duplication of work being conducted by the Government Accountability Office (GAO), congressional committees, and executive branch agency and independent commission investigations.\n\n		Reports\n\nEach proposed commission would be tasked with issuing a final report detailing its findings, conclusions, and recommendations. H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 would provide that the commission "may submit" interim reports to Congress and the President, but do not provide time lines on when those reports might be submitted. In each case, the interim report would need to be agreed to by a majority of commission members. H.R. 6431 would also require the commission to submit a report on actions taken by the states and a report on essential products, materials, ingredients, and equipment required to fight pandemics.\nH.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 also specify that final reports shall be agreed to by a majority of commission members. H.R. 6455 does not specify a vote threshold for approval of its report.\nNone of the bills make specific provisions for the inclusion of minority viewpoints. Presumably this would leave each commission with discretion on whether to include or exclude minority viewpoints. Past advisory entities have been proposed or established with a variety of statutory reporting conditions, including the specification of majority or super-majority rules for report adoption and provisions requiring the inclusion of minority viewpoints. In practice, advisory bodies that are not given statutory direction on these matters have tended to work under simple-majority rules for report adoption.\n\n		Report Deadlines\n\nH.R. 6429 would require a final report one year after the commission\'s initial meeting. H.R. 6431 and H.R. 6440 would require a final report not later than 18 months after enactment. H.R. 6455 would require a final report to be published not later than 18 months after the commission\'s first meeting. \nH.R. 6548 would require a final report by October 15, 2021. This deadline could be extended by 90 days upon a vote of no fewer than 8 (out of 10) commission members. The commission could vote to extend its final report deadline up to three times, and would be required to notify Congress, the President, and the public of any such extension.\nWhile such a deadline would potentially give the commission a defined period of time to complete its work, setting a particular date for report completion could potentially create unintended time constraints. Any delay in the passage of the legislation or in the appointment process would reduce the amount of time the commission has to complete its work, even with the opportunity for the commission to extend its own deadline up to three times.\nThe length of time a congressional commission has to complete its work is arguably one of the most consequential decisions when designing an advisory entity. If the entity has a short window of time, the quality of its work product may suffer or it may not be able to fulfill its statutory mandate on time.\nOn the other hand, if the commission is given a long period of time to complete its work, it may undermine one of a commission\'s primary legislative advantages, the timely production of expert advice on a current matter. A short deadline may also affect the process of standing up a new commission. The selection of commissioners, recruitment of staff, arrangement of office space, and other logistical matters may require expedited action if short deadlines need to be met.\n\n		Report Submission\n\nOf the five proposed commissions, four ( H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6548 ) are directed to submit their reports to both Congress and the President. H.R. 6455 requires that the report is made public.\nMost congressional advisory commissions are required to submit their reports to Congress, and sometimes to the President or an executive department or agency head. For example, the National Commission on Severely Distressed Public Housing\'s final report was submitted to both Congress and the Secretary of Housing and Urban Development.\n\n		Commission Termination\n\nCongressional commissions are usually statutorily mandated to terminate. Termination dates for most commissions are linked to either a fixed period of time after the establishment of the commission, the selection of members, or the date of submission of the commission\'s final report. Alternatively, some commissions are given fixed calendar termination dates.\nAll five commission proposals would provide for the commission to terminate within a certain period of time following submission of its final report. H.R. 6429 , H.R. 6431 , H.R. 6440 , and H.R. 6455 would each direct the commission to terminate 60 days after the submission; H.R. 6548 specifies a time line of 90 days after submission.\n\n	Commission Powers\n\nEach of the five proposals would provide the proposed commission with certain powers to carry out its mission (see Table 1 for specifics). One general issue for commissions is who is authorized to execute such powers. In some cases, the commission itself executes its powers, with the commission deciding whether to devise rules and procedures for the general use of such power. In other cases, the legislation specifically authorizes the commission to give discretionary power to subcommittees or individual commission members. Finally, the legislation itself might grant certain powers to individual members of the commission, such as the chair.\n\n		Hearings and Evidence\n\nAll five bills would provide the proposed commission with the power to hold hearings, take testimony, and receive evidence. All five commissions would also be provided the power to administer oaths to witnesses.\n\n		Subpoenas\n\nFour of the bills would provide the commission with subpoena power. H.R. 6440 would not provide subpoena power to the commission. H.R. 6429 , H.R. 6431 , and H.R. 6548 would provide that subpoenas could only be issued by either (1) agreement of the chair and vice chair, or (2) the affirmative vote of 6 (of 10) commission members. H.R. 6455 would require that a subpoena could only be issued by either agreement of the chair and vice chair or an affirmative vote of 18 (of 25) commission members. All four bills that would provide subpoena power contain substantially similar judicial methods of subpoena enforcement.\n\n		Administrative Support\n\nAll five of the bills would provide that the commission receive administrative support from the General Services Administration (GSA). The GSA provides administrative support to dozens of federal entities, including congressional advisory commissions. Each of the five bills would provide that GSA be reimbursed for its services by the commission. Each bill also provides that other departments or agencies may provide funds, facilities, staff, and other services to the commission.\n\n		Other Powers\n\nWithout explicit language authorizing certain activities, commissions often cannot gather information, enter into contracts, use the U.S. mail like an executive branch entity, or accept donations or gifts. \nAll five bills direct that federal agencies provide information to the commission upon request. H.R. 6429 , H.R. 6431 , and H.R. 6548 would also provide that the commission could use the U.S. mails in the same manner as any department or agency, enter into contracts, and accept gifts or donations of services or property.\n\n	Staffing\n\nThe proposed COVID-19 commissions contain staffing provisions commonly found in congressional advisory commission legislation. Congressional advisory commissions are usually authorized to hire staff. Most statutes specify that the commission may hire a lead staffer, often referred to as a "staff director," "executive director," or another similar title, in addition to additional staff as needed. Rather than mandate a specific staff size, many commissions are instead authorized to appoint a staff director and other personnel as necessary, subject to the limitations of available funds.\nMost congressional commissions are also authorized to hire consultants, procure intermittent services, and request that federal agencies detail personnel to aid the work of the commission.\n\n		Director and Commission Staff\n\nFour of the bills provide that the commission may hire staff without regard to certain laws regarding the competitive service; H.R. 6440 does not specifically exempt the commission from such laws. Four bills ( H.R. 6429 , H.R. 6431 , H.R. 6455 , and H.R. 6548 ) would authorize, but not require, the commission to hire a staff director and additional staff, as appropriate. Four proposals would limit staff salaries to level V of the executive schedule. Three of the bills would specifically designate staff as federal employees for the purposes of certain laws, such as workman\'s compensation, retirement, and other benefits.\n\n		Detailees\n\nWhen authorized, some commissions can have federal agency staff detailed to the commission. All five bills would provide that federal employees could be detailed to the commission. Four bills would provide that the detailee would be without reimbursement to his or her home agency. H.R. 6440 would allow detailees on a reimbursable basis. \n\n		Experts and Consultants\n\nAll five bills would provide the commission with the authority to hire experts and consultants. Four of the bills limit the rate of pay for consultants to level IV of the Executive Schedule. H.R. 6440 does not specify a specific limit.\n\n		Security Clearances\n\nFour bills would provide that federal agencies and departments shall cooperate with the commission to provide members and staff appropriate security clearances. H.R. 6440 does not contain a security clearance provision.\n\n	Funding and Costs\n\nCommissions generally require funding to help meet their statutory goals. When designing a commission, therefore, policymakers may consider both how the commission will be funded, and how much funding the commission will be authorized to receive. Four of the five proposals specify a funding mechanism for the commission.\nHow commissions are funded and the amounts that they receive vary considerably. Several factors can contribute to overall commission costs. These factors might include the cost of hiring staff, contracting with outside consultants, and engaging administrative support, among others. Additionally, most commissions reimburse the travel expenditures of commissioners and staff, and some compensate their members. The duration of a commission can also significantly affect its cost; past congressional commissions have been designed to last anywhere from several months to several years.\n\n		Costs\n\nIt is difficult to estimate or predict the potential overall cost of any commission. Annual budgets for congressional advisory entities range from several hundred thousand dollars to millions of dollars annually. Overall expenses for any individual advisory entity depend on a variety of factors, the most important of which are the number of paid staff and the commission\'s duration and scope. Some commissions have few full-time staff; others employ large numbers, such as the National Commission on Terrorist Attacks Upon the United States, which had a full-time paid staff of nearly 80. Secondary factors that can affect commission costs include the number of commissioners, how often the commission meets or holds hearings, whether or not the commission travels or holds field hearings, and the publications the commission produces.\n\n		Authorized Funding\n\nThree of the bills ( H.R. 6429 , H.R. 6440 , and H.R. 6548 ) would authorize the appropriation of "such sums as may be necessary" for the commission, to be derived in equal amounts from the contingent fund of the Senate and the applicable accounts of the House of Representatives. H.R. 6429 and H.R. 6548 would provide that funds are available until the commission terminates. H.R. 6455 would authorize the appropriation of $4 million for the commission, to remain available until the commission terminates. H.R. 6431 does not include an authorization of appropriations.\n\n	Comparison of Proposals to Create a COVID-19 Commission\n\n Table 1 provides a side-by-side comparison of major provisions of the five proposals. For each bill, the membership structure, appointment structure, rules of procedure and operation, duties and reporting requirements, proposed commission powers, staffing provisions, and funding are compared.\n\nSummary:\n'
DEFAULT_PROMPTS.append(long_prompt)

NUM_TOP_LOGPROBS = 5


def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    if torch_dtype is torch.float32:
        return "float32"
    else:
        raise NotImplementedError()


def get_top_logprobs(logits, k):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs, top_indices = torch.topk(logprobs, k=k, dim=-1)
    return logprobs


def get_token_ids_logprobs(logits, token_ids):
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    del logits
    logprobs = logprobs[..., token_ids]
    return logprobs


def _get_sentence_transformer_embedding_model(
    model_path, torch_dtype, matryoshka_dim: Optional[int] = None
):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import is_sentence_transformer_model

    if is_sentence_transformer_model(model_path):
        model = SentenceTransformer(
            model_path,
            model_kwargs={"torch_dtype": torch_dtype},
            truncate_dim=matryoshka_dim,
        )
    else:  # if no pre-trained sentence-transformers model
        from sentence_transformers import models

        word_embedding_model = models.Transformer(model_path).to(dtype=torch_dtype)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode="lasttoken",
        )
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], truncate_dim=matryoshka_dim
        )

    return model.cuda()


@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    top_output_logprob_idx: List[List[int]] = None
    embed_logits: List[torch.Tensor] = None
    scores: List[float] = None
    input_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    output_token_logprobs_lst: List[List[Tuple[float, int, None]]] = None
    token_ids_input_logprobs: List[torch.Tensor] = None
    token_ids_output_logprobs: List[torch.Tensor] = None


class HFRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str = "generation",
        output_str_only: bool = False,
        trust_remote_code: bool = False,
        patch_model_do_sample_false: bool = False,
        matryoshka_dim: Optional[int] = None,
    ):
        self.model_type = model_type
        self.output_str_only = output_str_only
        self.trust_remote_code = trust_remote_code
        self.patch_model_do_sample_false = patch_model_do_sample_false

        self.in_queue = mp.Queue()
        self.out_queue = mp.Queue()

        self.model_proc = mp.Process(
            target=self.start_model_process,
            args=(
                self.in_queue,
                self.out_queue,
                model_path,
                torch_dtype,
                matryoshka_dim,
            ),
        )
        self.model_proc.start()

    def needs_trust_remote_code(self, model_path):
        models_needs_trust_remote = [
            "LxzGordon/URM-LLaMa-3.1-8B",
        ]
        if model_path in models_needs_trust_remote:
            return True
        return False

    # copy from https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py

    def _get_gme_qwen2_vl_embeddings(
        self, prompts, image_data: Optional[List[str]] = None
    ):

        images = None
        if image_data is not None:
            images = [load_image(image)[0] for image in image_data]

        inputs = self.processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=1800,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = self._forward_gme_qwen2_vl(**inputs)
        return embeddings.tolist()

    def _forward_gme_qwen2_vl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).to(inputs_embeds.device)
                image_mask = input_ids == self.model.config.image_token_id
                inputs_embeds[image_mask] = image_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
        )

        embeddings = outputs.hidden_states[-1][:, -1]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def start_model_process(
        self,
        in_queue,
        out_queue,
        model_path,
        torch_dtype,
        matryoshka_dim: Optional[int] = None,
    ):
        # Apply model-specific patches
        monkey_patch_gemma2_sdpa()

        # Load the model and tokenizer
        if self.model_type == "generation":
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=self.trust_remote_code
            )
            if self.trust_remote_code:
                model_cls = AutoModelForCausalLM
            else:
                model_arch = getattr(config, "architectures")[0]
                model_cls = getattr(transformers, model_arch)
            self.base_model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
                low_cpu_mem_usage=True,
            ).cuda()
        elif self.model_type == "embedding":
            if "gme-qwen2-vl" in model_path.lower():
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                ).cuda()
                self.processor = AutoProcessor.from_pretrained(model_path)
            elif "clip" in model_path.lower():
                self.model = AutoModel.from_pretrained(model_path).cuda()
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                self.model = _get_sentence_transformer_embedding_model(
                    model_path, torch_dtype, matryoshka_dim=matryoshka_dim
                )
        elif self.model_type == "reward" or self.model_type == "cross_encoder":
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.needs_trust_remote_code(model_path),
            ).cuda()
        else:
            raise Exception(f"Unrecognized model type {self.model_type}")
        self.tokenizer = get_tokenizer(
            model_path,
            torch_dtype=torch.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        # Run forward
        while True:
            prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob = (
                in_queue.get()
            )
            if lora_paths is not None:
                assert len(prompts) == len(lora_paths)

            if prompts is not None:
                if self.model_type == "generation":
                    out_queue.put(
                        self.forward_generation_raw(
                            base_model=self.base_model,
                            prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            tokenizer=self.tokenizer,
                            lora_paths=lora_paths,
                            torch_dtype=torch_dtype,
                            output_str_only=self.output_str_only,
                            token_ids_logprob=token_ids_logprob,
                            patch_model_do_sample_false=self.patch_model_do_sample_false,
                        )
                    )
                elif self.model_type == "embedding":
                    assert not self.output_str_only
                    if "gme-qwen2-vl" in model_path.lower():
                        logits = self._get_gme_qwen2_vl_embeddings(prompts, image_data)
                    elif "clip" in model_path.lower():
                        if image_data is not None:
                            image = load_image(image_data)
                            inputs = self.processor(
                                images=image[0], return_tensors="pt"
                            )
                            logits = self.model.get_image_features(
                                pixel_values=inputs.data["pixel_values"].cuda(),
                            ).tolist()
                        else:
                            inputs = self.tokenizer(
                                prompts, padding=True, return_tensors="pt"
                            )
                            logits = self.model.get_text_features(
                                input_ids=inputs.data["input_ids"].cuda(),
                                attention_mask=inputs.data["attention_mask"].cuda(),
                            ).tolist()
                    else:
                        logits = self.model.encode(prompts).tolist()
                    out_queue.put(ModelOutput(embed_logits=logits))
                elif self.model_type == "cross_encoder":
                    inputs = self.tokenizer(
                        prompts, padding=True, return_tensors="pt"
                    ).to("cuda")
                    scores = self.model(**inputs).logits
                    scores = scores.squeeze().tolist()
                    if not isinstance(scores, list):
                        scores = [scores]
                    out_queue.put(ModelOutput(scores=scores))

                elif self.model_type == "reward":
                    scores = []
                    for conv in prompts:
                        conv_formatted = self.tokenizer.apply_chat_template(
                            conv, tokenize=False
                        )
                        conv_tokenized = self.tokenizer(
                            conv_formatted, return_tensors="pt"
                        ).to("cuda")
                        scores.append(
                            float(self.model(**conv_tokenized).logits[0][0].item())
                        )
                    out_queue.put(ModelOutput(scores=scores))
                else:
                    raise Exception(f"Unrecognized model type {self.model_type}")

    def forward(
        self,
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        token_ids_logprob: Optional[int] = None,
    ):
        self.in_queue.put(
            (prompts, image_data, max_new_tokens, lora_paths, token_ids_logprob)
        )
        return self.out_queue.get()

    def terminate(self):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_proc.terminate()
        self.in_queue = self.out_queue = None

    @staticmethod
    def forward_generation_raw(
        base_model,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int,
        tokenizer,
        torch_dtype: torch.dtype,
        lora_paths: Optional[List[str]] = None,
        output_str_only: bool = False,
        token_ids_logprob: Optional[int] = None,
        patch_model_do_sample_false: Optional[bool] = False,
    ) -> ModelOutput:
        output_strs = []
        top_input_logprobs = []
        top_output_logprobs = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        for i, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = tokenizer.encode(p, return_tensors="pt").cuda()
            else:
                input_ids = torch.tensor([p], device="cuda")

            if lora_paths is not None and lora_paths[i] is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    base_model,
                    lora_paths[i],
                    torch_dtype=torch_dtype,
                    is_trainable=False,
                )
            else:
                model = base_model
            if patch_model_do_sample_false:
                model.generation_config.do_sample = False
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=(not output_str_only),
                    # make sure to disable compile
                    disable_compile=True,
                ),
            )

            text = tokenizer.decode(
                outputs[0][0][len(input_ids[0]) :], skip_special_tokens=True
            )
            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)

            if not output_str_only:
                # outputs.scores: (num_token, 1, vocab_size)
                top_output_logprobs.append(
                    [
                        get_top_logprobs(logits[0], NUM_TOP_LOGPROBS).tolist()
                        for logits in outputs.scores
                    ]
                )
                if token_ids_logprob is not None:
                    token_ids_output_logprobs.append(
                        [
                            get_token_ids_logprobs(
                                logits[0], token_ids_logprob
                            ).tolist()
                            for logits in outputs.scores
                        ]
                    )
                del outputs

                input_logits = model.forward(input_ids).logits[0]
                top_input_logprobs.append(
                    get_top_logprobs(input_logits, NUM_TOP_LOGPROBS).tolist()
                )
                if token_ids_logprob is not None:
                    token_ids_input_logprobs.append(
                        get_token_ids_logprobs(input_logits, token_ids_logprob).tolist()
                    )
                del input_logits

            if lora_paths is not None and lora_paths[i] is not None:
                # Unload the LoRA adapter if it is used
                model.unload()

        return ModelOutput(
            output_strs=output_strs,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )


class SRTRunner:
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype,
        model_type: str,
        tp_size: int = 1,
        model_impl: str = "auto",
        port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        lora_paths: Optional[Union[List[str], List[dict[str, str]]]] = None,
        max_loras_per_batch: int = 4,
        attention_backend: Optional[str] = None,
        prefill_attention_backend: Optional[str] = None,
        decode_attention_backend: Optional[str] = None,
        lora_backend: str = "csgmv",
        disable_cuda_graph: bool = False,
        disable_radix_cache: bool = False,
        chunked_prefill_size: Optional[int] = None,
        dp_size: int = 1,
        tokenizer_path: Optional[str] = None,
        mem_fraction_static: float = 0.65,
        trust_remote_code: bool = False,
        speculative_draft_model_path: Optional[str] = None,
        speculative_draft_model_revision: Optional[str] = None,
        speculative_algorithm: Optional[str] = None,
        speculative_num_steps: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
        disable_overlap_schedule: bool = False,
        disable_custom_all_reduce: bool = False,
        torchao_config: Optional[str] = None,
        cuda_graph_max_bs: int = 4,
        sleep_on_idle=False,
        max_lora_rank: Optional[int] = None,
        lora_target_modules: Optional[List[str]] = None,
        enable_lora: Optional[bool] = None,
        max_loaded_loras: Optional[int] = None,
        json_model_override_args: Optional[dict[str, Any]] = None,
        lora_eviction_policy: str = "lru",
    ):
        self.model_type = model_type
        self.is_generation = model_type == "generation"
        enable_dp_attention = dp_size > 1

        spec_kwargs = {}
        if speculative_draft_model_path:
            spec_kwargs["speculative_draft_model_path"] = speculative_draft_model_path
            spec_kwargs["speculative_draft_model_revision"] = (
                speculative_draft_model_revision
            )
            spec_kwargs["speculative_algorithm"] = speculative_algorithm
            spec_kwargs["speculative_num_steps"] = speculative_num_steps
            spec_kwargs["speculative_eagle_topk"] = speculative_eagle_topk
            spec_kwargs["speculative_num_draft_tokens"] = speculative_num_draft_tokens

        self.engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            model_impl=model_impl,
            torchao_config=torchao_config,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=trust_remote_code,
            is_embedding=not self.is_generation,
            lora_paths=lora_paths,
            max_loras_per_batch=max_loras_per_batch,
            lora_backend=lora_backend,
            attention_backend=attention_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            disable_cuda_graph=disable_cuda_graph,
            disable_radix_cache=disable_radix_cache,
            chunked_prefill_size=chunked_prefill_size,
            enable_dp_attention=enable_dp_attention,
            dp_size=dp_size,
            tokenizer_path=tokenizer_path,
            disable_overlap_schedule=disable_overlap_schedule,
            cuda_graph_max_bs=cuda_graph_max_bs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            sleep_on_idle=sleep_on_idle,
            max_lora_rank=max_lora_rank,
            lora_target_modules=lora_target_modules,
            enable_lora=enable_lora,
            max_loaded_loras=max_loaded_loras,
            json_model_override_args=(
                json.dumps(json_model_override_args)
                if json_model_override_args
                else "{}"
            ),
            lora_eviction_policy=lora_eviction_policy,
            **spec_kwargs,
        )

        if tokenizer_path is None:
            self.tokenizer = get_tokenizer(
                model_path, trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = None

    def load_lora_adapter(self, lora_name: str, lora_path: str, pinned: bool = False):
        return self.engine.load_lora_adapter(lora_name, lora_path, pinned)

    def unload_lora_adapter(self, lora_name: str):
        return self.engine.unload_lora_adapter(lora_name)

    def forward(
        self,
        prompts: Union[
            List[List[str]], List[str], List[torch.Tensor]
        ] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
        dimensions: Optional[int] = None,
    ):
        if self.is_generation:
            return self.forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths,
                logprob_start_len=logprob_start_len,
                top_k=top_k,
                token_ids_logprob=token_ids_logprob,
            )
        else:
            if self.model_type == "embedding":
                response = self.engine.encode(
                    prompt=prompts, image_data=image_data, dimensions=dimensions
                )
                if isinstance(response, list):
                    logits = [x["embedding"] for x in response]
                else:
                    logits = [response["embedding"]]
                return ModelOutput(embed_logits=logits)
            # cross encoder model
            elif self.model_type == "cross_encoder":
                response = self.engine.rerank(prompts)
                if not isinstance(response, list):
                    response = [response]
                scores = [x["embedding"] for x in response]
                return ModelOutput(scores=scores)
            # reward model
            else:
                response = self.engine.encode(prompts)
                scores = [x["embedding"][0] for x in response]
                return ModelOutput(scores=scores)

    def batch_forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        image_data: Optional[List[str]] = None,
        max_new_tokens=8,
        lora_paths=None,
    ):
        """
        testing serving by sending all prompts once
        only return output strings and no logprobs
        """
        if self.is_generation:
            return self.batch_forward_generation_raw(
                engine=self.engine,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                lora_paths=lora_paths,
            )
        else:
            response = self.engine.encode(prompts, image_data)
            if self.model_type == "embedding":
                logits = [x["embedding"] for x in response]
                return ModelOutput(embed_logits=logits)
            else:
                scores = [x["embedding"][0] for x in response]
                return ModelOutput(scores=scores)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.shutdown()
        del self.engine

    @staticmethod
    def forward_generation_raw(
        engine: Engine,
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens: int = 8,
        lora_paths: Optional[List[str]] = None,
        logprob_start_len: int = 0,
        top_k: Optional[int] = None,
        token_ids_logprob: Optional[List[int]] = None,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        output_ids = []
        # Input logprobs. Note that the last item in input logprob is equivalent to
        # the first item in the output logprob.
        top_input_logprobs = []
        input_token_logprobs_lst = []
        top_output_logprobs = []
        output_token_logprobs_lst = []
        top_output_logprob_idx = []
        if token_ids_logprob is not None:
            token_ids_input_logprobs = []
            token_ids_output_logprobs = []
        else:
            token_ids_input_logprobs = token_ids_output_logprobs = None

        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        if top_k:
            sampling_params["top_k"] = top_k

        for i, prompt in enumerate(prompts):
            response = engine.generate(
                prompt,
                lora_path=lora_paths[i] if lora_paths else None,
                sampling_params=sampling_params,
                return_logprob=True,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=NUM_TOP_LOGPROBS,
                token_ids_logprob=token_ids_logprob,
            )
            text = response["text"]

            # Check if the text is empty or only whitespace.
            if not text.strip():
                raise ValueError(
                    "Received an empty text response. Please verify your input or model configuration."
                )
            output_strs.append(text)
            # output_ids.append(response["output_ids"])

            input_token_logprobs = response["meta_info"]["input_token_logprobs"]
            output_token_logprobs = response["meta_info"]["output_token_logprobs"]
            # print(i, input_token_logprobs)
            # print(i, output_token_logprobs)
            logprobs = response["meta_info"]["input_top_logprobs"]
            if token_ids_logprob is not None:
                input_token_ids_logprobs = response["meta_info"][
                    "input_token_ids_logprobs"
                ][1:]
            else:
                input_token_ids_logprobs = None

            num_prompt_tokens = response["meta_info"]["prompt_tokens"]
            assert len(input_token_logprobs) == num_prompt_tokens - logprob_start_len
            assert len(logprobs) == num_prompt_tokens - logprob_start_len

            # The first token logprob has no meaning in sglang.
            input_token_logprobs = input_token_logprobs[1:]
            logprobs = logprobs[1:]
            assert len(input_token_logprobs) == len(logprobs)

            input_token_logprobs_lst.append(
                input_token_logprobs + [output_token_logprobs[0]]
            )
            output_token_logprobs_lst.append(output_token_logprobs)

            top_input_logprobs.append(
                [[tup[0] for tup in x[:NUM_TOP_LOGPROBS]] for x in logprobs]
                + [
                    [
                        tup[0]
                        for tup in response["meta_info"]["output_top_logprobs"][0][
                            :NUM_TOP_LOGPROBS
                        ]
                    ]
                ]
            )
            top_output_logprobs.append(
                [
                    [tup[0] for tup in x[:NUM_TOP_LOGPROBS]]
                    for x in response["meta_info"]["output_top_logprobs"]
                ]
            )
            top_output_logprob_idx.append(
                [
                    [tup[1] for tup in x[:NUM_TOP_LOGPROBS]]
                    for x in response["meta_info"]["output_top_logprobs"]
                ]
            )
            if token_ids_logprob is not None:
                token_ids_input_logprobs.append(
                    [[tup[0] for tup in x] for x in input_token_ids_logprobs]
                    + [
                        [
                            tup[0]
                            for tup in response["meta_info"][
                                "output_token_ids_logprobs"
                            ][0]
                        ]
                    ]
                )
                token_ids_output_logprobs.append(
                    [
                        [tup[0] for tup in x]
                        for x in response["meta_info"]["output_token_ids_logprobs"]
                    ]
                )

        return ModelOutput(
            output_strs=output_strs,
            output_ids=output_ids,
            top_input_logprobs=top_input_logprobs,
            top_output_logprobs=top_output_logprobs,
            input_token_logprobs_lst=input_token_logprobs_lst,
            output_token_logprobs_lst=output_token_logprobs_lst,
            top_output_logprob_idx=top_output_logprob_idx,
            token_ids_input_logprobs=token_ids_input_logprobs,
            token_ids_output_logprobs=token_ids_output_logprobs,
        )

    @staticmethod
    def batch_forward_generation_raw(
        prompts: Union[List[str], List[torch.Tensor]],
        max_new_tokens,
        lora_paths,
        engine,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}
        response = engine.generate(
            prompts,
            lora_path=lora_paths if lora_paths else None,
            sampling_params=sampling_params,
        )
        output_strs = [r["text"] for r in response]

        return ModelOutput(
            output_strs=output_strs,
        )


def monkey_patch_gemma2_sdpa():
    """
    Use sdpa by default to fix the OOM issue.
    Revert this commit:
    https://github.com/huggingface/transformers/commit/975b988bfe6e7ebb47390cd9a1556c6888804883#diff-5f76eac6f18f4b491521314c318a9692318feb4d19228e9576cce7bde4240834R660
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2PreTrainedModel

    def _check_and_enable_sdpa(config, hard_check_only: bool = False):
        config._attn_implementation = "sdpa"
        return config

    setattr(Gemma2PreTrainedModel, "_check_and_enable_sdpa", _check_and_enable_sdpa)


def check_close_model_outputs(
    hf_outputs: ModelOutput,
    srt_outputs: ModelOutput,
    prefill_tolerance: float,
    decode_tolerance: float,
    rouge_l_tolerance: float,
    debug_text: str = "",
    check_logprobs: bool = True,
):
    # Compare output strings
    print(f"{hf_outputs.output_strs=}")
    print(f"{srt_outputs.output_strs=}")
    rouge_l_scores = calculate_rouge_l(hf_outputs.output_strs, srt_outputs.output_strs)
    print(f"{rouge_l_scores=}")
    assert all(
        score >= rouge_l_tolerance for score in rouge_l_scores
    ), f"Not all ROUGE-L scores are greater than rouge_l_tolerance={rouge_l_tolerance}"

    if check_logprobs:
        for i in range(len(hf_outputs.output_strs)):
            # Compare input logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_input_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_input_logprobs[i])
            input_len = hf_logprobs.shape[0]
            print(
                "prefill logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < prefill_tolerance), (
                    f"prefill logprobs are not all close with {debug_text} "
                    f"prefill_tolerance={prefill_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )

            # Compare output logprobs
            hf_logprobs = torch.Tensor(hf_outputs.top_output_logprobs[i])
            srt_logprobs = torch.Tensor(srt_outputs.top_output_logprobs[i])

            print(
                "decode logprobs max_diff", torch.max(abs(hf_logprobs - srt_logprobs))
            )
            if input_len <= 100:
                assert torch.all(abs(hf_logprobs - srt_logprobs) < decode_tolerance), (
                    f"decode logprobs are not all close with {debug_text} "
                    f"decode_tolerance={decode_tolerance}."
                    f"{hf_logprobs=}, {srt_logprobs=}"
                )
