package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.config.RagProperties;
import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.model.DocumentVisibility;
import com.hypersale.raglogin.model.KnowledgeDocument;
import com.hypersale.raglogin.repository.KnowledgeDocumentRepository;
import com.hypersale.raglogin.service.DocumentAccessService;
import com.hypersale.raglogin.service.RagClient;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.UUID;

@RestController
@RequestMapping("/api/documents")
public class DocumentController {
    private static final Set<String> SUPPORTED = Set.of("md", "markdown", "txt", "html", "htm", "json", "pdf", "docx", "pptx", "xlsx");
    private final KnowledgeDocumentRepository documents;
    private final RagProperties properties;
    private final DocumentAccessService access;
    private final RagClient rag;

    public DocumentController(KnowledgeDocumentRepository documents, RagProperties properties, DocumentAccessService access, RagClient rag) {
        this.documents = documents; this.properties = properties; this.access = access; this.rag = rag;
    }

    @GetMapping
    public List<?> list(@AuthenticationPrincipal AppUser user) {
        return access.accessible(user).stream().map(document -> java.util.Map.of(
                "id", document.getId(), "name", document.getOriginalName(), "visibility", document.getVisibility().name(),
                "status", document.getIndexStatus(), "owner_user_id", document.getOwnerUserId(), "created_at", document.getCreatedAt()
        )).toList();
    }

    @PostMapping(consumes = "multipart/form-data")
    public Object upload(@AuthenticationPrincipal AppUser user,
                         @RequestParam("file") MultipartFile file,
                         @RequestParam(defaultValue = "PRIVATE") DocumentVisibility visibility,
                         @RequestParam(required = false, defaultValue = "") String grantedUserIds) {
        if (file.isEmpty()) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "文件不能为空");
        String originalName = Path.of(file.getOriginalFilename() == null ? "upload" : file.getOriginalFilename()).getFileName().toString();
        String extension = originalName.contains(".") ? originalName.substring(originalName.lastIndexOf('.') + 1).toLowerCase() : "";
        if (!SUPPORTED.contains(extension)) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "不支持的文档类型");
        if (visibility == DocumentVisibility.TENANT_PUBLIC && user.getRole().name().equals("USER")) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "普通用户不能发布租户公共文档");
        }
        try {
            String documentId = UUID.randomUUID().toString();
            Path tenantRoot = access.tenantRoot(user.getTenantId(), properties.getDocumentRoot());
            Files.createDirectories(tenantRoot);
            Path stored = tenantRoot.resolve(documentId + "-" + originalName).normalize();
            if (!stored.startsWith(tenantRoot)) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "文件名无效");
            file.transferTo(stored);
            KnowledgeDocument document = new KnowledgeDocument();
            document.setId(documentId); document.setTenantId(user.getTenantId()); document.setOwnerUserId(user.getId());
            document.setOriginalName(originalName); document.setStoredPath(stored.toAbsolutePath().toString()); document.setVisibility(visibility);
            if (!grantedUserIds.isBlank()) {
                document.setGrantedUserIds(Arrays.stream(grantedUserIds.split(",")).map(String::trim).filter(value -> value.matches("\\d+")).map(Long::valueOf).collect(java.util.stream.Collectors.toSet()));
            }
            KnowledgeDocument saved = documents.save(document);
            try { rag.ingestTenant(user.getTenantId()); saved.setIndexStatus("INDEXED"); }
            catch (RuntimeException exception) { saved.setIndexStatus("PENDING"); }
            documents.save(saved);
            return java.util.Map.of("id", saved.getId(), "name", saved.getOriginalName(), "status", saved.getIndexStatus(), "message", "文件已上传，索引正在更新");
        } catch (java.io.IOException exception) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "文件保存失败", exception);
        }
    }
}
