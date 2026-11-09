package com.hypersale.raglogin.controller;

import com.hypersale.raglogin.model.AppUser;
import com.hypersale.raglogin.repository.KnowledgeDocumentRepository;
import com.hypersale.raglogin.service.DocumentAccessService;
import com.hypersale.raglogin.service.RagClient;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.util.Map;

@RestController
@RequestMapping("/api/documents")
public class DocumentPreviewController {
    private final KnowledgeDocumentRepository documents;
    private final DocumentAccessService access;
    private final RagClient rag;

    public DocumentPreviewController(KnowledgeDocumentRepository documents, DocumentAccessService access, RagClient rag) {
        this.documents = documents; this.access = access; this.rag = rag;
    }

    @GetMapping("/{id}/preview")
    public Map<String, Object> preview(@AuthenticationPrincipal AppUser user, @PathVariable String id,
                                       @RequestParam(defaultValue = "1") int lineStart,
                                       @RequestParam(defaultValue = "1") int lineEnd) {
        var document = documents.findByTenantIdAndId(user.getTenantId(), id)
                .filter(item -> access.canRead(user, item))
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "文档不存在或无权访问"));
        try { return rag.preview(user.getTenantId(), document.getStoredPath(), lineStart, lineEnd); }
        catch (RagClient.RagUnavailableException exception) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, exception.getMessage());
        }
    }
}
