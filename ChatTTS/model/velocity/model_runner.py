import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .configs import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.logger import init_logger
from .model_loader import get_model
from vllm.model_executor import InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast,
    broadcast_object_list,
)
from .sampling_params import SamplingParams, SamplingType
from .sequence import (
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
    SequenceGroupOutput,
    SequenceOutput,
)
from vllm.utils import in_wsl
from ..embed import Embed
from .sampler import Sampler
from safetensors.torch import safe_open

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]


class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        is_driver_worker: bool = False,
        post_model_path: str = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_driver_worker = is_driver_worker
        self.post_model_path = post_model_path

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (
            model_config.get_sliding_window() if model_config is not None else None
        )
        self.model = None
        self.block_size = None  # Set after initial profiling.

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None
            else 0
        )
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()

        from ...config import Config
        self.config = Config()

        from ..speaker import Speaker
        device = None
        if device is None:
            device = select_device()
            self.logger.info("use device %s", str(device))

        self.speaker = Speaker(
            self.config.gpt.hidden_size, self.config.spk_stat, device
        )
        # spek_sta = "愐穤巩噅廷戇笉屈癐媄垹垧帶爲漈塀殐慄亅倴庲舴猂瑈圐狴夥圓帍戛挠腉耐劤坽喳幾战謇聀崒栄呥倸庭燡欈杁襐褄乭埗幺爃弔摁斐捔兕佖廐舏竾豃磐姓趡佄幒爚欄豄讐皳訵仩帆投謌荃蝐叄圝伆幦抂茁呄掑斃讹傮庞爣蜀橁偐祄亥兡常爂欍扉丐浔佱僈強払伅扂蛐徴憍傞巀戺欀艂琐嗴啥値彷刂權穈扒卤俔贲庛初笂卄贐枴仭亁庛剎猢扃缐趤刁偵幪舏伌煁婐潤晍位弾舙茥穁葏蠣訑企庤刊笍橁溑僔云偁庯戚伍潉膐脴僵噔廃艅匊祂唐憴壝嗙席爥欁虁谐牴帽势弿牳蜁兀蛐傄喩丿帔刔圆衁廐罤庁促帙劢伈汄樐檄勵伴弝舑欍罅虐昴劭勅帜刼朊蕁虐蓴樑伫幨扑謪剀堐稴丵伱弐舮諸赁習俔容厱幫牶謃孄糐答嗝僊帜燲笄終瀒判久僤帘爴茇千孑冄凕佳引扐蜁歁缏裄剽儺恘爋朏眿廐呄塍嘇幻爱茠詁訐剴唭俐幾戊欀硁菐贄楕偒巡爀弎屄莐睳賙凶彎刅漄區唐溴剑劋庽舽猄煃跐夔惥伾庮舎伈罁垑坄怅业怯刁朇獁嶏覔坩俳巶爜朐潁崐萄俹凛常爺笌穀聐此夡倛帡刀匉終窏舣販侽怿扉伥贿憐忓謩姆幌犊漂慆癒却甝兎帼戏欅詂浐朔仹壭帰臷弎恇菐獤帡偖帘爞伅腂皐纤囅充幓戠伥灂丐訤戱倱弋爮嬌癁恐孄侥劬忶刓國詀桒古偩嘄庬戚茝赂监燤嘑勌幦舽持呂諐棤姑再底舡笍艃瀐孴倉傔弋爔猠乁濑塄偽嘧恂舛缇襃厐窴仡刱忕別漇穁岏缴廽价庌爊謈硄讑惤倁儂庭爋伇蝂嶐莔摝傠库刞茄歃戏薤伍伯廮创笠塄熐兴勽俄帅剉最腀砐敤卝侍弆戺朒虃旐蚄梕亖幔牻朣扅贐玔堝噅帡剌圅摀崐彤流僳庙爖嬇啁渐悤堁丛幆刧挜彃悐幤刹嚟恕芁看聀摐焔向乁帖爭欁癃糒圄弙佱廜戤謍婀咐昴焍亩廦艏拼謿芐癤怹兽幸舳朇畁喐稔毝丼弈懲挀譂勑哴啁伎常舭笯晁堑俄叩剔廟爍欦絁夒伤休傑廳戌蜅潆癐彴摑勯床刽欅艁砐忄搉从廡舊猥潂唐委仱僜廼爤朄呃弐礔滵垓幩爄挂筁乐籤刕凟幵爠弉癅乑吴勥伖帪舩茆婁碐幤叭乢巜艳猁桀桐啄唩俊幍舮猀艅焐螔琽亀帋爜缅噃咐斤喩予幩爛笆摀浐猴依侹幃刕園慄蛐栤澹仑座爼謉桃慐浔斕偻幛懰嬓衁愐氄悅仿应芔漄衃敐謤傁匩幹抃圉癄廐裄屵噉幍利謍聂搐蛔嚙坍怗舁圐畃膐栄刵东巆戤諾呃偑媤嗨跞忶爝眄祂朒嶔僭劉忾刐匋癄袐翴珅僷廲芄茈恈皐擄崑伄廉牍匃剃犏澤唑丄庺戃伃煀某杄偙亽帴切缌罄挐尴噙倰带舞漄橄塐糴俩僯帀般漀坂栐更両俇廱舌猁慂拐偤嶱卶应刪眉獁茐伔嘅偺帟舊漂恀栐暄喡乞庙舆匂敀潑恔劑侖延戦盽怶唯慳蝘蟃孫娎益袰玍屃痶翮笪儚裀倹椌玻翀詵筽舘惯堿某侰晈藏缮詗廦夸妎瑻瀒裔媀憞唃冶璭狻渠荑奬熹茅愺氰菣滠翦岓褌泣崲嚭欓湒聙宺爄蛅愸庍匃帆誔穮懌蓪玷澌氋抌訙屌臞廛玸听屺希疭孝凂紋新煎彃膲跱尪懁眆窴珏卓揨菸紭概囥显壌榄垫嘮嬭覤媸侵佮烒耸觌婀秋狃帹葯訤桜糨笾腢伀肶悍炂艤禖岅臺惘梷瞍友盁佨岧憳瓧嘴汬藊愌蘤嶠硴绤蜲襏括勾谂縨妥蓪澭竭萢藜纞糲煮愆瀯孯琓罂諺塿燗狟弙衯揻縷丱糅臄梱瀮杰巳猙亊符胠匃泀廏圃膂蒃籏礩岈簹缌劺燲褡孓膜拔蠿觮呋煣厌尷熜論弲牭紫寊誃紀橴賬傸箍弚窃侫簲慯烣渽祌壓媥噜夽夛諛玹疮禄冪謇媽衤盰缺繑薫兾萧嵱打滽箺嚯凣狢蠜崼覽烸簶盯籓摀苶峸懗泲涻凮愳緗剋笔懆廡瞿椏礤惐藥崍腈烄伹亯昣翬褍絋桫僨吨莌丛矄蜞娈憊苆塁蓏嚢嫼绻崱婋囱蠸篯晣芀繼索兓僖誹岯圪褰蠇唓妷胅巁渮砛傈蝷嵚冃購赁峍裋荂舾符熻岳墩寮粃凲袑彚太绲头摯繳狁俥籌冝諝註坎幫擤詒宒凕賐唶梎噔弼課屿覍囨焬櫱撪蝮蝬簸懰櫫涺嵍睻屪翔峞慘滟熲昱军烊舿尦舄糖奁溏凂彆蝲糴禍困皻灏牋睒诙嶱臀开蓈眎腼丢纻廏憤嫖暭袭崲肸螛妒榗紉谨窮袃瑠聍绊腆亿冲葐喋縔詖岑兾给堸赏旻桀蛨媆訂峦紷敯囬偐筨岸焸拭笵殒哜墒萍屓娓諙械臮望摰芑寭准僞谹氍旋憢菮屃划欣瘫谎蘻哐繁籥禦僿誵皯墓燀縿笞熦绗稹榎矻綞蓓帡戓沺区才畃洊詪糐裶盰窶耎偌劂誐庩惝滜沺哮呃煐譠崄槀猄肼蔐擋湌蠺篃恥諌瞦宍堫挪裕崑慩狲悠煋仛愞砈粵八棁害楐妋萔貨尵奂苰怫誎傫岆蕯屇脉夈仆茎刓繸芺壸碗曛汁戭炻獻凉媁兎狜爴怰賃纎袏娷禃蓥膹薪渻罸窿粫凾褄舺窮墫干苊繁冏僮訸夯绛蓪虛羽慲烏憷趎睊蠰莍塞成廎盁欏喓蜮譤崆楁囘矇薭伣艘虝帴奮苢渶虎暣翐蝃尾稈糶瀴罐嵚氮葯笫慐棌悶炯竻爅们媡姢嫺窷刮歫劈裩屬椕賑蜹薊刲義哯尗褦瓀稾礋揣窼舫尋姁椄侸嗫珺修纘媃腽蛛稹梭呛瀈蘟縀礉論夵售主梮蠉娅娭裀誼嶭観枳倊簈褃擞綿催瞃溶苊笛襹櫲盅六囫獩佃粨慯瓢眸旱荃婨蔞岋祗墼焻网牻琖詆峋秉胳媴袭澓賢経稟壩胫碯偏囫嶎纆窈槊賐撹璬莃缘誾宭愊眗喷监劋萘訯總槿棭戾墮犄恌縈簍樥蛔杁袭嫛憫倆篏墵賈羯茎觳蒜致娢慄勒覸蘍曲栂葭宆妋皽缽免盳猼蔂糥觧烳檸佯憓煶蔐筼种繷琲膌塄剰讎対腕棥渽忲俛浪譬秛惛壒嘸淫冻曄睻砃奫貯庴爅粓脮脡娎妖峵蘲討惋泊蠀㴆" 
        # self.speaker = Speaker(
        #     768, spek_sta, 'cuda:0'
        # )

    def load_model(self) -> None:
        self.model = get_model(self.model_config)
        self.post_model = Embed(
            self.model_config.get_hidden_size(),
            self.model_config.num_audio_tokens,
            self.model_config.num_text_tokens,
        )
        state_dict_tensors = {}
        with safe_open(self.post_model_path, framework="pt", device=0) as f:
            for k in f.keys():
                state_dict_tensors[k] = f.get_tensor(k)
        self.post_model.load_state_dict(state_dict_tensors)
        self.post_model.to(next(self.model.parameters())).eval()
        self.sampler = Sampler(self.post_model, self.model_config.num_audio_tokens, 4)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (
            self.max_context_len_to_capture + block_size - 1
        ) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32
        )

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []

        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)
            use_refine = seq_data.use_refine
            spk_emb = seq_data.spk_emb
            text_mask = seq_data.text_mask

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prompt_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        max_prompt_len = max(prompt_lens)
        input_tokens = _make_tensor_with_pad(
            input_tokens, max_prompt_len, pad=0, dtype=torch.long
        )
        input_positions = _make_tensor_with_pad(
            input_positions, max_prompt_len, pad=0, dtype=torch.long
        )
        slot_mapping = _make_tensor_with_pad(
            slot_mapping, max_prompt_len, pad=_PAD_SLOT_ID, dtype=torch.long
        )

        input_metadata = InputMetadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            use_cuda_graph=False,
        )
        return input_tokens, input_positions, input_metadata, prompt_lens, use_refine, spk_emb, text_mask

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                context_len = (
                    seq_len
                    if self.sliding_window is None
                    else min(seq_len, self.sliding_window)
                )
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                if self.sliding_window is not None:
                    sliding_window_blocks = self.sliding_window // self.block_size
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        batch_size = len(input_tokens)
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture
        )
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(1)
                block_tables.append([])
            batch_size = graph_batch_size

        input_tokens = _make_tensor_with_pad(
            input_tokens, max_len=1, pad=0, dtype=torch.long, device="cuda"
        )
        input_positions = _make_tensor_with_pad(
            input_positions, max_len=1, pad=0, dtype=torch.long, device="cuda"
        )
        slot_mapping = _make_tensor_with_pad(
            slot_mapping, max_len=1, pad=_PAD_SLOT_ID, dtype=torch.long, device="cuda"
        )
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, : len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device="cuda")
        else:
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_context_len,
                pad=0,
                dtype=torch.int,
                device="cuda",
            )

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        max_prompt_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[sampling_params.sampling_type].append(
                    categorized_sample_indices_start_idx
                )
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(
                            selected_token_start_idx,
                            selected_token_start_idx + prompt_len - 1,
                        )
                    )
                selected_token_indices.append(selected_token_start_idx + prompt_len - 1)
                selected_token_start_idx += max_prompt_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx, selected_token_start_idx + num_seqs)
                )
                selected_token_start_idx += num_seqs

                categorized_sample_indices[sampling_params.sampling_type].extend(
                    range(
                        categorized_sample_indices_start_idx,
                        categorized_sample_indices_start_idx + num_seqs,
                    )
                )
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_h2d(
            selected_token_indices, dtype=torch.long, pin_memory=not self.in_wsl
        )
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, pin_memory=not self.in_wsl)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata]:
        use_refine = False
        spk_emb = None
        text_mask = None
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, input_metadata, prompt_lens, use_refine, spk_emb, text_mask) = (
                    self._prepare_prompt(seq_group_metadata_list)
                )
            else:
                (input_tokens, input_positions, input_metadata) = self._prepare_decode(
                    seq_group_metadata_list
                )
                prompt_lens = []
            sampling_metadata = self._prepare_sample(
                seq_group_metadata_list, prompt_lens
            )

            def get_size_or_none(x: Optional[torch.Tensor]):
                return x.size() if x is not None else None

            # Broadcast the input data. For input tensors, we first broadcast
            # its shape and then broadcast the tensor to avoid high
            # serialization cost.
            py_data = {
                "input_tokens_size": input_tokens.size(),
                "input_positions_size": input_positions.size(),
                "is_prompt": input_metadata.is_prompt,
                "slot_mapping_size": get_size_or_none(input_metadata.slot_mapping),
                "max_context_len": input_metadata.max_context_len,
                "context_lens_size": get_size_or_none(input_metadata.context_lens),
                "block_tables_size": get_size_or_none(input_metadata.block_tables),
                "use_cuda_graph": input_metadata.use_cuda_graph,
                "selected_token_indices_size": sampling_metadata.selected_token_indices.size(),
            }
            broadcast_object_list([py_data], src=0)
            # TODO(zhuohan): Combine the broadcasts or set async_op=True.
            broadcast(input_tokens, src=0)
            broadcast(input_positions, src=0)
            if input_metadata.slot_mapping is not None:
                broadcast(input_metadata.slot_mapping, src=0)
            if input_metadata.context_lens is not None:
                broadcast(input_metadata.context_lens, src=0)
            if input_metadata.block_tables is not None:
                broadcast(input_metadata.block_tables, src=0)
            broadcast(sampling_metadata.selected_token_indices, src=0)
        else:
            receving_list = [None]
            broadcast_object_list(receving_list, src=0)
            py_data = receving_list[0]
            input_tokens = torch.empty(
                *py_data["input_tokens_size"], dtype=torch.long, device="cuda"
            )
            broadcast(input_tokens, src=0)
            input_positions = torch.empty(
                *py_data["input_positions_size"], dtype=torch.long, device="cuda"
            )
            broadcast(input_positions, src=0)
            if py_data["slot_mapping_size"] is not None:
                slot_mapping = torch.empty(
                    *py_data["slot_mapping_size"], dtype=torch.long, device="cuda"
                )
                broadcast(slot_mapping, src=0)
            else:
                slot_mapping = None
            if py_data["context_lens_size"] is not None:
                context_lens = torch.empty(
                    *py_data["context_lens_size"], dtype=torch.int, device="cuda"
                )
                broadcast(context_lens, src=0)
            else:
                context_lens = None
            if py_data["block_tables_size"] is not None:
                block_tables = torch.empty(
                    *py_data["block_tables_size"], dtype=torch.int, device="cuda"
                )
                broadcast(block_tables, src=0)
            else:
                block_tables = None
            selected_token_indices = torch.empty(
                *py_data["selected_token_indices_size"], dtype=torch.long, device="cuda"
            )
            broadcast(selected_token_indices, src=0)
            input_metadata = InputMetadata(
                is_prompt=py_data["is_prompt"],
                slot_mapping=slot_mapping,
                max_context_len=py_data["max_context_len"],
                context_lens=context_lens,
                block_tables=block_tables,
                use_cuda_graph=py_data["use_cuda_graph"],
            )
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                perform_sampling=False,
            )

        return input_tokens, input_positions, input_metadata, sampling_metadata, use_refine, spk_emb, text_mask

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        input_tokens, input_positions, input_metadata, sampling_metadata, use_refine, spk_emb, text_mask = (
            self.prepare_input_tensors(seq_group_metadata_list)
        )
        # print(sampling_metadata.seq_data)
        seq_groups = []
        input_tokens_history = []
        for i, rtn in enumerate(sampling_metadata.seq_groups):
            seq_groups.append(rtn[0][0])
            tokens_history = sampling_metadata.seq_data[rtn[0][0]].output_token_ids
            if len(tokens_history) >= 1:
                if len(tokens_history[0]) == 1:
                    tokens_history = [token[0] for token in tokens_history]
                else:
                    tokens_history = [list(token) for token in tokens_history]
            input_tokens_history.append(tokens_history)
        input_tokens_history = torch.tensor(input_tokens_history).to(
            input_tokens.device
        )
        # token_ids = rtn.outputs[0].token_ids
        # for j, token_id in enumerate(token_ids):
        #     if len(token_id) == 1:
        #         token_ids[j] = token_id[0]
        #     else:
        #         token_ids[j] = list(token_id)

        # Execute the model.
        # print("it1",input_tokens)
        if len(input_tokens.shape) == 2:
            input_tokens = input_tokens.unsqueeze(2).repeat(1, 1, 4)
        if len(input_tokens_history.shape) == 2:
            input_tokens_history = input_tokens_history.unsqueeze(2).repeat(1, 1, 4)
        # print(input_tokens_history.shape)
        # print("it2",input_tokens.shape)
        # text_mask = input_tokens != 0
        # text_mask = text_mask[:, :, 0]
        if text_mask is None:
            text_mask = input_tokens != 0
            text_mask = text_mask[:, :, 0]

        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        infer_text = sampling_metadata.seq_groups[0][1].infer_text
        temperture = sampling_metadata.seq_groups[0][1].temperature
        if not infer_text:
            temperture = torch.tensor(temperture).to(input_tokens.device)
        logits_processors, logits_warpers = sampling_metadata.seq_groups[0][
            1
        ].logits_processors
        # print(logits_processors, logits_warpers)
        min_new_token = sampling_metadata.seq_groups[0][1].min_new_token
        eos_token = sampling_metadata.seq_groups[0][1].eos_token
        start_idx = sampling_metadata.seq_groups[0][1].start_idx
        if input_tokens.shape[-2] == 1:
            if infer_text:
                input_emb: torch.Tensor = self.post_model.emb_text(
                    input_tokens[:, :, 0]
                )
            else:
                code_emb = [
                    self.post_model.emb_code[i](input_tokens[:, :, i])
                    for i in range(self.post_model.num_vq)
                ]
                input_emb = torch.stack(code_emb, 3).sum(3)
                start_idx = (
                    input_tokens_history.shape[-2] - 1
                    if input_tokens_history.shape[-2] > 0
                    else 0
                )
        else:
            input_emb = self.post_model(input_tokens, text_mask)
            if not use_refine:
                if spk_emb is not None:
                    self.speaker.apply(
                        input_emb,
                        spk_emb,
                        input_tokens,
                        21143,
                        'cuda:0',
                    )

        # print(input_emb.shape)
        hidden_states = model_executable(
            input_emb=input_emb,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )
        # print(hidden_states.shape)
        # print(input_tokens)
        B_NO_PAD = input_tokens_history.shape[0]
        input_tokens = input_tokens[:B_NO_PAD, :, :]
        hidden_states = hidden_states[:B_NO_PAD, :, :]
        idx_next, logprob, finish = self.sampler.sample(
            inputs_ids=(
                input_tokens
                if input_tokens_history.shape[-2] == 0
                else input_tokens_history
            ),
            hidden_states=hidden_states,
            infer_text=infer_text,
            temperature=temperture,
            logits_processors=logits_processors,
            logits_warpers=logits_warpers,
            min_new_token=min_new_token,
            now_length=1,
            eos_token=eos_token,
            start_idx=start_idx,
        )
        # print(logprob.shape, idx_next.shape)
        if len(logprob.shape) == 2:
            logprob = logprob[:, None, :]
        logprob = torch.gather(logprob, -1, idx_next.transpose(-1, -2))[:, :, 0]
        # print("测试",idx_next.shape, logprob.shape)
        # Sample the next token.
        # output = self.model.sample(
        #     hidden_states=hidden_states,
        #     sampling_metadata=sampling_metadata,
        # )
        results = []
        for i in range(idx_next.shape[0]):
            idx_next_i = idx_next[i, 0, :].tolist()
            logprob_i = logprob[i].tolist()
            tmp_hidden_states = hidden_states[i]
            if input_tokens[i].shape[-2] != 1:
                tmp_hidden_states = tmp_hidden_states[-1:, :]
            result = SequenceGroupOutput(
                samples=[
                    SequenceOutput(
                        parent_seq_id=seq_groups[i],
                        logprobs={tuple(idx_next_i): logprob_i},
                        output_token=tuple(idx_next_i),
                        hidden_states=tmp_hidden_states,
                        finished=finish[i].item(),
                    ),
                ],
                prompt_logprobs=None,
            )
            results.append(result)
        # print(results)
        # print(idx_next, idx_next.shape, logprob.shape)
        return results

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(
            top_p=0.99, top_k=vocab_size - 1, infer_text=True
        )
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = max_num_batched_tokens // max_num_seqs + (
                group_id < max_num_batched_tokens % max_num_seqs
            )
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        assert not self.model_config.enforce_eager
        logger.info(
            "Capturing the model for CUDA graphs. This may lead to "
            "unexpected consequences if the model is not static. To "
            "run the model in eager mode, set 'enforce_eager=True' or "
            "use '--enforce-eager' in the CLI."
        )
        logger.info(
            "CUDA graphs can take additional 1~3 GiB memory per GPU. "
            "If you are running out of memory, consider decreasing "
            "`gpu_memory_utilization` or enforcing eager mode."
        )
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_emb = torch.zeros(
            max_batch_size,
            1,
            self.model_config.get_hidden_size(),
            dtype=next(self.model.parameters()).dtype,
        ).cuda()
        input_positions = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(_BATCH_SIZES_TO_CAPTURE):
            # Create dummy input_metadata.
            input_metadata = InputMetadata(
                is_prompt=False,
                slot_mapping=slot_mapping[:batch_size],
                max_context_len=self.max_context_len_to_capture,
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_emb[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                input_metadata,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_emb: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_emb,
            positions,
            kv_caches,
            input_metadata,
        )
        torch.cuda.synchronize()

        # Capture the graph.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states = self.model(
                input_emb,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_emb": input_emb,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_emb: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_emb"].copy_(input_emb, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(
            input_metadata.slot_mapping, non_blocking=True
        )
        self.input_buffers["context_lens"].copy_(
            input_metadata.context_lens, non_blocking=True
        )
        self.input_buffers["block_tables"].copy_(
            input_metadata.block_tables, non_blocking=True
        )

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    if len(x) == max_len:
        return list(x)
    return list(x) + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    padded_x = []
    for x_i in x:
        pad_i = pad
        if isinstance(x[0][0], tuple):
            pad_i = (0,) * len(x[0][0])
        padded_x.append(_pad_to_max(x_i, max_len, pad_i))

    return torch.tensor(
        padded_x,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory and str(device) == "cpu",
    )


def _get_graph_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8


def _async_h2d(data: list, dtype, pin_memory):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device="cuda", non_blocking=True)
